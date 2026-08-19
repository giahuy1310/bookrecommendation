"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
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
    <section>
      <h1>User profile</h1>
      <p>User id: {userId ?? "—"}</p>
      <button type="button" onClick={handleLogout}>
        Log out
      </button>
    </section>
  );
}
