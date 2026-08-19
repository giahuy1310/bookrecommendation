"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { isLoggedIn } from "../lib/auth";

type AuthGuardProps = {
  requireAuth: boolean;
  redirectTo: string;
  children: React.ReactNode;
};

export default function AuthGuard({
  requireAuth,
  redirectTo,
  children,
}: AuthGuardProps) {
  const router = useRouter();
  const [allowed, setAllowed] = useState(false);

  useEffect(() => {
    const loggedIn = isLoggedIn();

    if (requireAuth && !loggedIn) {
      router.replace(redirectTo);
      return;
    }

    if (!requireAuth && loggedIn) {
      router.replace(redirectTo);
      return;
    }

    setAllowed(true);
  }, [requireAuth, redirectTo, router]);

  if (!allowed) return null;

  return <>{children}</>;
}
