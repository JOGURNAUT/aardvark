import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Evidence } from "./App";
import type { Snippet } from "./types";

// These render rather than reduce, because the crash that made this file exist
// was not in the reducer. `select_done` carries `chars`, a length, and the
// evidence list called `.slice()` on a `text` field that is never on the wire.
// The reducer stored the snippets happily; the render threw, React unmounted
// the whole tree, and the page went black with nothing on it naming the cause.
//
// renderToStaticMarkup rather than jsdom: it throws on exactly the same render
// error, needs no DOM, and keeps the front-end suite at one dependency.

const wire = (over: Partial<Snippet> = {}): Snippet => ({
  // Exactly the fields loop.py puts in select_done. No `text`.
  idx: 1,
  title: "Annex III of the AI Act",
  domain: "eur-lex.europa.eu",
  url: "https://eur-lex.europa.eu/x",
  score: 0.847,
  chars: 1284,
  ...over,
});

describe("Evidence", () => {
  it("renders a snippet shaped exactly like the wire", () => {
    const html = renderToStaticMarkup(<Evidence snippets={[wire()]} />);
    expect(html).toContain("Annex III of the AI Act");
    expect(html).toContain("eur-lex.europa.eu");
    expect(html).toContain("1,284 characters");
    expect(html).toContain("0.847");
  });

  it("does not throw when the snippet has no text field", () => {
    // The regression itself. A required `text: string` in types.ts made this
    // compile and blank the page at runtime.
    expect(() => renderToStaticMarkup(<Evidence snippets={[wire()]} />)).not.toThrow();
  });

  it("survives a snippet missing every optional field", () => {
    // types.ts is hand-written against a docstring, so it can be wrong in the
    // other direction too. Nothing optional may be dereferenced blind.
    const bare = { title: "", domain: "", url: "https://x.test/a" } as Snippet;
    const html = renderToStaticMarkup(<Evidence snippets={[bare]} />);
    expect(html).toContain("https://x.test/a");
  });

  it("falls back to the url when there is no title", () => {
    const html = renderToStaticMarkup(
      <Evidence snippets={[wire({ title: "", url: "https://x.test/b" })]} />,
    );
    expect(html).toContain("https://x.test/b");
  });

  it("shows the text when it is present, without assuming it is", () => {
    const html = renderToStaticMarkup(
      <Evidence snippets={[wire({ text: "the operative sentence" })]} />,
    );
    expect(html).toContain("the operative sentence");
  });

  it("renders every snippet, numbering them from one", () => {
    const html = renderToStaticMarkup(
      <Evidence snippets={[wire({ idx: 1 }), wire({ idx: 2, title: "Second" })]} />,
    );
    expect(html).toContain("Second");
    expect(html).toContain("2 snippets");
  });
});
