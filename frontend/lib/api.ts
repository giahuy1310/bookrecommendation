export type Pick = {
  isbn: string;
  title: string;
  author: string;
  finalScore: number;
};

export type TopPicksResponse = {
  userId: number;
  contextIsbn: string | null;
  picks: Pick[];
};

export type BookRow = {
  isbn: string;
  title: string;
  author: string;
};

export type InteractionEvent = {
  userId: number;
  isbn: string;
  eventType: "READ" | "ADD_TO_CART" | "ADD_TO_COLLECTION";
  createdAtMs: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function getTopPicks(userId: number): Promise<TopPicksResponse> {
  return request<TopPicksResponse>(
    `/api/top-picks?userId=${encodeURIComponent(String(userId))}`
  );
}

export async function searchBooks(
  q: string,
  limit = 20
): Promise<BookRow[]> {
  return request<BookRow[]>(
    `/api/search?q=${encodeURIComponent(q)}&limit=${encodeURIComponent(String(limit))}`
  );
}

export async function postInteraction(
  body: InteractionEvent
): Promise<void> {
  await request<unknown>("/api/interactions", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
