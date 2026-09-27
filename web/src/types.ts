// Mirrors the event shapes documented at the top of agent/loop.py.
//
// These are hand-written rather than generated. A generator would be better,
// but the loop is the source of truth and it is a Python docstring, so the
// honest description is: this file can drift, and the discriminated union is
// what stops a drift from being silent. An unknown `type` falls through the
// switch in App.tsx and is counted, not ignored.

export interface PlanEvent {
  type: "plan";
  queries: string[];
  strategy: string;
}

export interface SearchDoneEvent {
  type: "search_done";
  results: { url: string; title: string; score?: number | null }[];
}

export interface FetchedPage {
  url: string;
  title: string;
  domain: string;
  ok?: boolean;
  chars?: number;
}

export interface FetchDoneEvent {
  type: "fetch_done";
  pages: FetchedPage[];
}

export interface Snippet {
  idx?: number;
  title: string;
  domain: string;
  url: string;
  score?: number;
  // The select_done event carries `chars`, a length, NOT the snippet text.
  // This was declared as a required `text: string` and the UI called
  // .slice() on it, so the first real query blanked the page: the exception
  // unmounted the tree and left a black screen with nothing in the console
  // that named the cause.
  //
  // Every field here is optional for the same reason. This file is written by
  // hand against a docstring in loop.py, so it can be wrong, and a type that
  // promises more than the wire delivers turns a missing field into a crash
  // instead of a blank.
  chars?: number;
  text?: string;
}

export interface SelectDoneEvent {
  type: "select_done";
  snippets: Snippet[];
}

export interface AnswerTokenEvent {
  type: "answer_token";
  text: string;
}

export interface Citation {
  marker: number;
  url: string;
  title: string;
  domain: string;
}

export interface AnswerDoneEvent {
  type: "answer_done";
  answer: string;
  citations: Citation[];
  latency_ms: number;
  stage_latencies: Record<string, number>;
  provider: string;
}

export interface ErrorEvent {
  type: "error";
  stage: string;
  message: string;
}

export type AgentEvent =
  | PlanEvent
  | SearchDoneEvent
  | FetchDoneEvent
  | SelectDoneEvent
  | AnswerTokenEvent
  | AnswerDoneEvent
  | ErrorEvent;

export type Stage = "plan" | "search" | "fetch" | "select" | "answer";

export const STAGES: Stage[] = ["plan", "search", "fetch", "select", "answer"];
