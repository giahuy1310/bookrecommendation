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
      <section className="inner-page padding-large">
        <div className="container">
          <div className="row justify-content-center">
            <div className="col-md-6 col-lg-5">
              <div className="card border rounded-3 p-4 p-md-5">
                <h1 className="mb-3">User profile</h1>
                <p className="mb-4">
                  User id: <span className="fw-bold">{userId ?? "—"}</span>
                </p>
                <button
                  type="button"
                  className="btn btn-dark w-100"
                  onClick={handleLogout}
                >
                  Log out
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </AuthGuard>
  );
}
