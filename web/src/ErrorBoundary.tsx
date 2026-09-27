import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

/**
 * Keeps a render crash from blanking the page.
 *
 * React unmounts the entire tree when a render throws, so the first real query
 * against this UI produced a black screen: `select_done` sends `chars`, a
 * length, and the evidence list called `.slice()` on a `text` field that was
 * never on the wire. The stack was in the console, but the page said nothing
 * at all, which is the worst way for a front end to fail.
 *
 * A boundary cannot prevent that bug. What it changes is that the failure
 * arrives with the error in it, on screen, next to a reload — and the rest of
 * the answer stays readable, since the boundary sits around the parts that
 * render server data rather than around the whole app.
 */
interface Props {
  children: ReactNode;
  label: string;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept so the component stack survives in the console even when the UI
    // only shows the message.
    console.error(`[${this.props.label}] render failed`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="error boundary">
        <span className="stage">{this.props.label}</span>
        <span className="msg">
          {this.state.error.message || String(this.state.error)}
          <button className="btn-ghost inline"
                  onClick={() => this.setState({ error: null })}>
            retry
          </button>
        </span>
      </div>
    );
  }
}
