const scheduleUrl = new URL("./schedule.html", document.baseURI);
scheduleUrl.search = window.location.search;
scheduleUrl.hash = window.location.hash;

const destination = `${scheduleUrl.pathname}${scheduleUrl.search}${scheduleUrl.hash}`;
const link = document.querySelector("#schedule-redirect-link");
if (link) link.href = destination;

window.location.replace(scheduleUrl.href);
