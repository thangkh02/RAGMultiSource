"use client";

import { useEffect, useState } from "react";

import { DocumentCard } from "@/components/documents/DocumentCard";
import { DocumentList } from "@/components/documents/DocumentList";
import { UploadBox } from "@/components/documents/UploadBox";
import { apiClient } from "@/lib/api-client";
import type { DocumentItem } from "@/features/documents/types";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const ownerUserId = "demo_user_001";

  async function refresh() {
    const result = await apiClient.documents.list(ownerUserId);
    setDocuments(result);
  }

  useEffect(() => {
    void refresh();
  }, []);

  return (
    <main className="page-shell">
      <div className="topbar">
        <div>
          <div className="brand">Documents</div>
          <div className="muted">Upload PDF or DOCX. The backend converts them to Markdown and chunks them.</div>
        </div>
      </div>
      <div className="grid">
        <section className="card stack">
          <UploadBox onUploaded={refresh} />
        </section>
        <section className="card">
          <h2>Uploaded documents</h2>
          <DocumentList documents={documents} renderItem={(doc) => <DocumentCard key={doc.id} document={doc} />} />
        </section>
      </div>
    </main>
  );
}
