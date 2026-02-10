fetch("navbar.html")
  .then(res => res.text())
  .then(html => {
    document.body.insertAdjacentHTML("afterbegin", html);

    const page = location.pathname.split("/").pop();
    document.querySelectorAll("nav a").forEach(a => {
      if (a.getAttribute("href") === page) {
        a.classList.add("active");
      }
    });
  });