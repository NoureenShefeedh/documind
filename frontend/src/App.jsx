import { useEffect, useState } from "react";
import { getToken, setToken } from "./api.js";
import AuthForm from "./components/AuthForm.jsx";
import Documents from "./components/Documents.jsx";
import Chat from "./components/Chat.jsx";

export default function App() {
  const [authed, setAuthed] = useState(Boolean(getToken()));
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    const onLogout = () => setAuthed(false);
    window.addEventListener("documind:logout", onLogout);
    return () => window.removeEventListener("documind:logout", onLogout);
  }, []);

  const logout = () => {
    setToken(null);
    setAuthed(false);
  };

  if (!authed) return <AuthForm onAuthed={() => setAuthed(true)} />;

  return (
    <div className="layout">
      <header>
        <h1>DocuMind</h1>
        <button className="link" onClick={logout}>Log out</button>
      </header>
      <main>
        <Documents refreshKey={refreshKey} onChanged={() => setRefreshKey((k) => k + 1)} />
        <Chat />
      </main>
    </div>
  );
}
