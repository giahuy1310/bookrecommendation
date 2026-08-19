import AuthGuard from "../../components/AuthGuard";

export default function CollectionPage() {
  return (
    <AuthGuard requireAuth redirectTo="/login">
      <section>
        <h1>My collection</h1>
        <p>Your saved books will appear here.</p>
      </section>
    </AuthGuard>
  );
}
