"use client";

import { useRouter } from "next/navigation";
import AuthGuard from "../../components/AuthGuard";
import { login } from "../../lib/auth";

export default function LoginPage() {
  const router = useRouter();

  function handleLogin() {
    login();
    router.push("/");
    router.refresh();
  }

  return (
    <AuthGuard requireAuth={false} redirectTo="/">
      <section>
        <h1>Log-in</h1>
        <p>Create a local numeric user id stored in this browser.</p>
        <button type="button" onClick={handleLogin}>
          Log-in
        </button>
      </section>
    </AuthGuard>
  );
}
