import AuthGuard from "../../components/AuthGuard";

export default function CartPage() {
  return (
    <AuthGuard requireAuth redirectTo="/login">
      <section>
        <h1>My cart</h1>
        <p>Books in your cart will appear here.</p>
      </section>
    </AuthGuard>
  );
}
