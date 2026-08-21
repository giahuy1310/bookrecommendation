"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import AuthGuard from "../../components/AuthGuard";
import { getUserId, logout } from "../../lib/auth";

export default function ProfilePage() {
  const router = useRouter();
  const [userId, setUserId] = useState<number | null>(null);

  useEffect(() => {
    setUserId(getUserId());
  }, []);

  function handleLogout() {
    logout();
    router.push("/login");
    router.refresh();
  }

  return (
    <AuthGuard requireAuth redirectTo="/login">
      <section>
        <h1>User profile</h1>
        <p>User id: {userId ?? "—"}</p>
        <button
          type="button"
          onClick={handleLogout}
          style={{
            padding: "8px 14px",
            borderRadius: 8,
            border: "1px solid var(--palette-blue)",
            background: "var(--palette-blue)",
            color: "var(--color-text)",
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          Log out
        </button>
      </section>
    </AuthGuard>
  );
}
