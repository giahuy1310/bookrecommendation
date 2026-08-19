import HorizontalPicksRow from "./HorizontalPicksRow";
import type { Pick } from "../lib/api";

type Props = {
  picks?: Pick[];
};

/** Thin placeholder; Task 3 wires fetch + arrow. */
export default function TopPicksSection({ picks = [] }: Props) {
  return (
    <section style={{ marginTop: 24 }}>
      <h2 style={{ marginBottom: 12 }}>Top picks</h2>
      <HorizontalPicksRow picks={picks} />
    </section>
  );
}
