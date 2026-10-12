import { getApiBase, ApiError } from "@/lib/api/client";
import { normalizeSources } from "@/lib/api/normalize";
import type { QuerySource } from "@/lib/types";

export interface StreamQueryOptions {
  query: string;
  conversationId?: string | null;
  collectionId?: string | null;
  signal?: AbortSignal;
  onToken: (token: string) => void;
  onSources: (sources: QuerySource[]) => void;
  onDone: (conversationId?: string) => void;
  onError: (error: Error) => void;
}

function parseSseChunk(
  chunk: string,
  handlers: Pick<
    StreamQueryOptions,
    "onToken" | "onSources" | "onDone" | "onError"
  >
) {
  const lines = chunk.split("\n");

  for (const line of lines) {
    if (!line.startsWith("data:")) continue;

    const data = line.slice(5).trim();

    if (!data) {
      continue;
    }

    if (data === "[DONE]") {
      handlers.onDone();
      return;
    }

    try {
      const parsed = JSON.parse(data) as Record<string, unknown>;
      const type = String(parsed.type ?? parsed.event ?? "");

      // ── Sources event ────────────────────────────────────────────────
      if (type === "sources" || parsed.sources) {
        handlers.onSources(
          normalizeSources(parsed.sources ?? parsed.data)
        );
        continue;
      }

      // ── Error event ─────────────────────────────────────────────────
      if (type === "error") {
        handlers.onError(
          new Error(
            String(parsed.message ?? parsed.content ?? "Stream error")
          )
        );
        return;
      }

      // ── Done event ──────────────────────────────────────────────────
      // Backend sends:
      // {
      //   "type": "done",
      //   "conversation_id": "<uuid>",
      //   "total_chars": N
      // }
      if (type === "done") {
        const conversationId =
          typeof parsed.conversation_id === "string"
            ? parsed.conversation_id
            : undefined;

        handlers.onDone(conversationId);
        return;
      }

      // ── Token event ─────────────────────────────────────────────────
      const token = String(
        parsed.content ??
          parsed.token ??
          parsed.text ??
          parsed.delta ??
          (type === "token" ? parsed.data : "") ??
          ""
      );

      if (token) {
        handlers.onToken(token);
      }
    } catch {
      // Non-JSON SSE data is treated as plain text.
      if (data) {
        handlers.onToken(data);
      }
    }
  }
}

export async function streamQuery(
  options: StreamQueryOptions
): Promise<void> {
  const apiBase = getApiBase();

  const res = await fetch(`${apiBase}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      query: options.query,
      conversation_id: options.conversationId ?? undefined,
      collection_id: options.collectionId ?? undefined,
      stream: true,
    }),
    signal: options.signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(res.status, text || res.statusText);
  }

  const contentType = res.headers.get("content-type") ?? "";

  // ── Non-streaming fallback ──────────────────────────────────────────
  if (!contentType.includes("text/event-stream")) {
    const json = (await res.json()) as Record<string, unknown>;

    if (json.sources) {
      options.onSources(normalizeSources(json.sources));
    }

    const answer = String(
      json.answer ?? json.response ?? json.content ?? ""
    );

    if (answer) {
      options.onToken(answer);
    }

    const conversationId =
      typeof json.conversation_id === "string"
        ? json.conversation_id
        : undefined;

    options.onDone(conversationId);
    return;
  }

  // ── Streaming response ──────────────────────────────────────────────
  if (!res.body) {
    throw new ApiError(500, "No response stream");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      parseSseChunk(part, options);
    }
  }

  if (buffer.trim()) {
    parseSseChunk(buffer, options);
  }

  // Safety fallback:
  // If the backend closes the stream without sending a done event,
  // still notify the UI that streaming has ended.
  options.onDone();
}