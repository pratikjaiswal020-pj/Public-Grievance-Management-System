const $ = (selector) => document.querySelector(selector);
const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map(item => item.msg).join("; ")
      : data.detail;
    throw new Error(detail || "Something went wrong. Please try again.");
  }
  return data;
};

const escapeHtml = (value = "") => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));
const displayDate = (value) => new Date(value).toLocaleString([], {dateStyle:"medium", timeStyle:"short"});

let predictionTimer;
let latestPrediction;

function issueChoices(result) {
  if (!result.multiple_issues_detected) return "";
  return `<p><strong>We found multiple issues. Confirm the tickets to create:</strong></p><div class="issue-options">${result.issues.map(issue => `<label><input type="checkbox" name="issue_categories" value="${escapeHtml(issue.category)}" checked> ${escapeHtml(issue.category)} <small>(${escapeHtml(issue.department)})</small></label>`).join("")}</div>`;
}

$("#description").addEventListener("input", (event) => {
  clearTimeout(predictionTimer);
  const text = event.target.value.trim();
  if (text.length < 20) return;
  predictionTimer = setTimeout(async () => {
    try {
      const result = await api("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text})});
      latestPrediction = result;
      const box = $("#prediction");
      box.hidden = false;
      box.innerHTML = `Likely route: <strong>${escapeHtml(result.department)}</strong> · ${escapeHtml(result.category)} · ${result.confidence}% confidence${result.priority === "High" ? " · <strong>High priority</strong>" : ""}${result.manual_review_recommended ? " · <strong>Manual review recommended</strong>" : ""}${issueChoices(result)}`;
    } catch (_) { /* Preview is optional; submitting still reports server errors. */ }
  }, 450);
});

$("#complaint-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button");
  button.disabled = true; button.textContent = "Submitting…";
  try {
    const formData = new FormData(form);
    const evidence = formData.get("evidence");
    formData.delete("evidence");
    const values = Object.fromEntries(formData);
    const selectedIssues = [...form.querySelectorAll('input[name="issue_categories"]:checked')].map(input => input.value);
    if (latestPrediction?.multiple_issues_detected && selectedIssues.length === 0) {
      throw new Error("Select at least one detected issue.");
    }
    if (selectedIssues.length) values.issue_categories = selectedIssues;
    const result = await api("/api/complaints", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(values)});
    if (evidence && evidence.size) {
      const upload = new FormData();
      upload.append("file", evidence);
      await api(`/api/complaints/${result.reference_id}/attachments`, {method:"POST", body:upload});
    }
    const submitted = $("#submitted");
    const tickets = result.related_complaints || [{reference_id: result.reference_id, category: result.category, department: result.department}];
    const ticketList = tickets.length > 1
      ? `<p><strong>Separate linked tickets were created:</strong></p><ul>${tickets.map(ticket => `<li><strong>${escapeHtml(ticket.category)}</strong> — ${escapeHtml(ticket.department)} (${escapeHtml(ticket.reference_id)})</li>`).join("")}</ul>`
      : `<p>It was routed to <strong>${escapeHtml(result.department)}</strong>.</p>`;
    submitted.hidden = false;
    submitted.innerHTML = `<h3>Complaint submitted</h3><p>Your main reference ID is:</p><p class="reference">${escapeHtml(result.reference_id)}</p>${ticketList}<p>Save this ID to track updates or attach evidence.</p>`;
    $("#reference-id").value = result.reference_id;
    form.reset(); latestPrediction = undefined; $("#prediction").hidden = true;
    submitted.scrollIntoView({behavior:"smooth", block:"center"});
  } catch (error) { alert(error.message); }
  finally { button.disabled = false; button.textContent = "Submit complaint"; }
});

function feedbackForm(referenceId) {
  return `<form class="feedback" data-feedback="${referenceId}"><strong>Rate this resolution</strong><p><select name="rating" aria-label="Rating"><option value="5">5 — Excellent</option><option value="4">4 — Good</option><option value="3">3 — Fair</option><option value="2">2 — Poor</option><option value="1">1 — Very poor</option></select></p><textarea name="comment" rows="3" placeholder="Optional feedback"></textarea><p><button class="button">Submit feedback</button></p></form>`;
}

$("#track-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const referenceId = $("#reference-id").value.trim().toUpperCase();
  const target = $("#tracking-result"); target.innerHTML = "";
  try {
    const item = await api(`/api/complaints/${encodeURIComponent(referenceId)}`);
    const related = (item.related_complaints || []).length > 1
      ? `<h4>Linked tickets</h4><ul>${item.related_complaints.map(ticket => `<li>${ticket.is_current ? "<strong>" : ""}${escapeHtml(ticket.category)} — ${escapeHtml(ticket.department)} (${escapeHtml(ticket.status)})${ticket.is_current ? "</strong>" : ""}</li>`).join("")}</ul>`
      : "";
    target.innerHTML = `<article class="card result"><span class="badge">${escapeHtml(item.status)}</span><h3>${escapeHtml(item.title)}</h3><p class="meta">${escapeHtml(item.reference_id)} · ${escapeHtml(item.category)} · ${escapeHtml(item.department)}</p><p>${escapeHtml(item.description)}</p>${related}<h4>Progress</h4><ol class="timeline">${item.timeline.map(event => `<li><strong>${escapeHtml(event.status)}</strong><p>${escapeHtml(event.remark)}</p><time>${displayDate(event.created_at)}</time></li>`).join("")}</ol>${item.attachments.length ? `<p><strong>Attachments:</strong> ${item.attachments.map(file => `<a href="${file.url}" target="_blank" rel="noopener">${escapeHtml(file.name)}</a>`).join(", ")}</p>` : ""}${item.status === "Resolved" && !item.feedback ? feedbackForm(item.reference_id) : ""}</article>`;
  } catch (error) { target.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`; }
});

document.addEventListener("submit", async (event) => {
  if (!event.target.matches("[data-feedback]")) return;
  event.preventDefault();
  const form = event.target;
  try {
    await api(`/api/complaints/${form.dataset.feedback}/feedback`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(Object.fromEntries(new FormData(form)))});
    form.innerHTML = "<strong>Thank you for your feedback.</strong>";
  } catch (error) { alert(error.message); }
});
