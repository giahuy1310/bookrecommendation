const SERVICES = [
  {
    icon: "cart-outline",
    title: "Free delivery",
    text: "Save books to your cart and collect them anytime.",
  },
  {
    icon: "quality",
    title: "Quality picks",
    text: "Recommendations tuned to your reading history.",
  },
  {
    icon: "price-tag",
    title: "Daily discovery",
    text: "Fresh suggestions as you search and interact.",
  },
  {
    icon: "shield-plus",
    title: "Local session",
    text: "Your numeric user id stays in this browser only.",
  },
];

export default function CompanyServices() {
  return (
    <section id="company-services" className="padding-large pb-0">
      <div className="container">
        <div className="row">
          {SERVICES.map((service) => (
            <div key={service.title} className="col-lg-3 col-md-6 pb-3 pb-lg-0">
              <div className="icon-box d-flex">
                <div className="icon-box-icon pe-3 pb-3">
                  <svg className={service.icon}>
                    <use xlinkHref={`#${service.icon}`} />
                  </svg>
                </div>
                <div className="icon-box-content">
                  <h4 className="card-title mb-1 text-capitalize text-dark">
                    {service.title}
                  </h4>
                  <p>{service.text}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
