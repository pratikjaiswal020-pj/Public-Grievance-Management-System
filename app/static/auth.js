/**
 * auth.js — shared JWT helpers for the CivicPulse portal.
 * Stores the token in localStorage under "cgms_token".
 */

const AUTH_KEY = "cgms_token";

export function getToken() {
  return localStorage.getItem(AUTH_KEY);
}

export function setToken(token) {
  localStorage.setItem(AUTH_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(AUTH_KEY);
}

export function parseToken(token) {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload;
  } catch {
    return null;
  }
}

export function isTokenValid(token) {
  if (!token) return false;
  const payload = parseToken(token);
  if (!payload) return false;
  return payload.exp * 1000 > Date.now();
}

export function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function getCurrentRole() {
  const token = getToken();
  if (!token || !isTokenValid(token)) return null;
  return parseToken(token)?.role ?? null;
}

export function getCurrentName() {
  const token = getToken();
  if (!token || !isTokenValid(token)) return null;
  return parseToken(token)?.name ?? null;
}

/**
 * Call at top of a protected page.
 * @param {"citizen"|"officer"|"admin"|string[]} required - role or list of allowed roles
 * @param {string} redirectUrl - where to redirect on failure
 */
export function requireAuth(required, redirectUrl = "/login.html") {
  const token = getToken();
  if (!token || !isTokenValid(token)) {
    window.location.replace(redirectUrl);
    return false;
  }
  const role = parseToken(token)?.role;
  const allowed = Array.isArray(required) ? required : [required];
  if (!allowed.includes(role)) {
    window.location.replace(redirectUrl);
    return false;
  }
  return true;
}

export function logout(redirectUrl = "/login.html") {
  clearToken();
  window.location.replace(redirectUrl);
}

/**
 * Shared API fetch wrapper that injects auth headers.
 */
export async function authApi(url, options = {}) {
  const headers = {
    ...(options.headers || {}),
    ...authHeaders(),
  };
  const response = await fetch(url, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join("; ")
      : data.detail;
    throw new Error(detail || "Something went wrong. Please try again.");
  }
  return data;
}
