"use client";

import { useState } from "react";

import { apiClient } from "@/lib/api-client";

type Props = {
  onUploaded: () => Promise<void>;
};

export function UploadBox({ onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const ownerUserId = "demo_user_001";
  const [busy, setBusy] = useState(false);

  return (
    <form
      className="stack"
      onSubmit={async (event) => {
        event.preventDefault();
        if (!file) return;
        setBusy(true);
        try {
          await apiClient.documents.upload(file, ownerUserId);
          await onUploaded();
          setFile(null);
        } finally {
          setBusy(false);
        }
      }}
    >
      <div className="field">
        <label>Upload PDF or DOCX</label>
        <input className="input" type="file" accept=".pdf,.docx" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
      </div>
      <button className="button" type="submit" disabled={!file || busy}>
        {busy ? "Uploading..." : "Upload"}
      </button>
    </form>
  );
}
