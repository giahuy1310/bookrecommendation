const USER_ID_KEY = "userId";

export function getUserId(): number | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_ID_KEY);
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function login(): number {
  const userId = Date.now() % 1_000_000_000;
  localStorage.setItem(USER_ID_KEY, String(userId));
  return userId;
}

export function logout(): void {
  localStorage.removeItem(USER_ID_KEY);
}
