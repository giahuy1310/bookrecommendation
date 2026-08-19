"use client";

import { useRouter } from "next/navigation";
import { login } from "../../lib/auth";

export default function LoginPage() {
  const router = useRouter();

  function handleLogin() {
    login();
    router.push("/");
    router.refresh();
  }

  return (
    <section>
      <h1>Log-in</h1>
      <p>Create a local numeric user id stored in this browser.</p>
      <button type="button" onClick={handleLogin}>
        Log-in
      </button>
    </section>
  );
}
