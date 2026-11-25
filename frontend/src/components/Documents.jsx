import { useEffect, useRef, useState } from "react";
import { deleteDocument, listDocuments, uploadDocument } from "../api.js";

export default function Documents({ refreshKey, onChanged }) {
  const [docs, setDocs] = useState([]);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const input = useRef(null);

  useEffect(() => {
    listDocuments().then(setDocs).catch((e) => setError(e.message));
  }, [refreshKey]);

  const onUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      await uploadDocument(file);
      onChanged();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      input.current.value = "";
    }
  };

  const remove = async (id) => {
    try {
      await deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <aside>
      <h2>Documents</h2>
      <label className="upload">
        {uploading ? "Indexing…" : "Upload .pdf, .txt or .md"}
        <input ref={input} type="file" accept=".pdf,.txt,.md" onChange={onUpload} disabled={uploading} hidden />
      </label>
      {error && <p className="error" role="alert">{error}</p>}
      <ul>
        {docs.map((d) => (
          <li key={d.id}>
            <span title={d.filename}>{d.filename}</span>
            <small>{d.n_chunks} chunks</small>
            <button className="link" onClick={() => remove(d.id)} aria-label={`Delete ${d.filename}`}>✕</button>
          </li>
        ))}
        {docs.length === 0 && <li className="muted">No documents yet.</li>}
      </ul>
    </aside>
  );
}
