const USER_ID_KEY = "userId";
const LOGGED_IN_KEY = "loggedIn";

function readStoredUserId(): number | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_ID_KEY);
  if (raw == null || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function isLoggedIn(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(LOGGED_IN_KEY) === "1";
}

/** Returns userId only when logged in (for API/nav). */
export function getUserId(): number | null {
  if (!isLoggedIn()) return null;
  return readStoredUserId();
}

export function login(): number {
  let userId = readStoredUserId();
  if (userId == null) {
    userId = Date.now() % 1_000_000_000;
    localStorage.setItem(USER_ID_KEY, String(userId));
  }
  localStorage.setItem(LOGGED_IN_KEY, "1");
  return userId;
}

export function logout(): void {
  localStorage.removeItem(LOGGED_IN_KEY);
}
