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
      <section className="inner-page padding-large">
        <div className="container">
          <div className="row justify-content-center">
            <div className="col-md-6 col-lg-5">
              <div className="card border rounded-3 p-4 p-md-5">
                <h1 className="mb-3">Log-in</h1>
                <p className="text-black-50">
                  Create a local numeric user id stored in this browser.
                </p>
                <button
                  type="button"
                  className="btn btn-dark w-100 my-3"
                  onClick={handleLogin}
                >
                  Log-in
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </AuthGuard>
  );
}
