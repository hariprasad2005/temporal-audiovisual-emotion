function SectionCard({ title, children, className = "" }) {
  return (
    <section className={`section-card ${className}`.trim()}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export default SectionCard;
