const BASE = import.meta.env.VITE_API_URL || "/api";
const TOKEN_KEY = "documind_token";

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY));

async function request(path, { method = "GET", json, body } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (json !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: json !== undefined ? JSON.stringify(json) : body,
  });

  if (res.status === 401 && token) {
    setToken(null);
    window.dispatchEvent(new Event("documind:logout"));
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

export const register = (email, password) => request("/auth/register", { method: "POST", json: { email, password } });
export const login = (email, password) => request("/auth/login", { method: "POST", json: { email, password } });
export const listDocuments = () => request("/documents");
export const deleteDocument = (id) => request(`/documents/${id}`, { method: "DELETE" });
export const uploadDocument = (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/documents", { method: "POST", body: form });
};
export const ask = (question) => request("/query", { method: "POST", json: { question } });
