import { getApiBase, ApiError } from "@/lib/api/client";
import { normalizeDocument } from "@/lib/api/normalize";
import type { Document } from "@/lib/types";

export interface UploadProgress {
  fileName: string;
  progress: number;
}

export async function uploadDocuments(
  files: File[],
  options?: {
    collectionId?: string | null;
    onProgress?: (progress: UploadProgress) => void;
    signal?: AbortSignal;
  }
): Promise<Document[]> {
  const results: Document[] = [];

  // Upload files one by one to avoid exceeding Next.js request body limit.
  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    if (options?.signal?.aborted) {
      throw new ApiError(0, "Upload cancelled");
    }

    const formData = new FormData();
    formData.append("files", file);

    if (options?.collectionId) {
      formData.append("collection_id", options.collectionId);
    }

    const result = await new Promise<Document>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const apiBase = getApiBase();

      xhr.open("POST", `${apiBase}/upload`);
      xhr.responseType = "json";

      // Set timeout to 15 minutes per file.
      xhr.timeout = 900000;

      const abortHandler = () => xhr.abort();

      if (options?.signal) {
        options.signal.addEventListener("abort", abortHandler, {
          once: true,
        });
      }

      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable || !options?.onProgress) return;

        const fileProgress = event.loaded / event.total;

        // Overall progress across all files.
        const overallProgress = Math.round(
          ((i + fileProgress) / files.length) * 100
        );

        options.onProgress({
          fileName: file.name,
          progress: overallProgress,
        });
      };

      xhr.onload = () => {
        if (options?.signal) {
          options.signal.removeEventListener("abort", abortHandler);
        }

        if (xhr.status >= 200 && xhr.status < 300) {
          const raw = xhr.response;

          if (Array.isArray(raw)) {
            const normalized = raw.map((item) =>
              normalizeDocument(item as Record<string, unknown>)
            );

            if (normalized.length > 0) {
              resolve(normalized[0]);
            } else {
              reject(new ApiError(500, "Upload returned no document"));
            }
            return;
          }

          const obj = (raw ?? {}) as Record<string, unknown>;

          const list =
            (obj.documents as unknown[]) ??
            (obj.results as unknown[]) ??
            (obj.items as unknown[]) ??
            (raw ? [raw] : []);

          if (list.length > 0) {
            resolve(
              normalizeDocument(list[0] as Record<string, unknown>)
            );
          } else {
            reject(new ApiError(500, "Upload returned no document"));
          }

          return;
        }

        const rawResponse = xhr.response;
        let errorMsg = xhr.statusText || "Upload failed";

        if (rawResponse && typeof rawResponse === "object") {
          const obj = rawResponse as Record<string, unknown>;

          if (typeof obj.detail === "string") {
            errorMsg = obj.detail;
          } else if (Array.isArray(obj.detail)) {
            errorMsg = obj.detail
              .map((d: any) => d?.msg ?? String(d))
              .join(", ");
          } else if (typeof obj.error === "string") {
            errorMsg = obj.error;
          } else if (typeof obj.message === "string") {
            errorMsg = obj.message;
          }
        }

        reject(new ApiError(xhr.status, errorMsg));
      };

      xhr.ontimeout = () => {
        if (options?.signal) {
          options.signal.removeEventListener("abort", abortHandler);
        }

        reject(
          new ApiError(
            408,
            `"${file.name}" upload timed out (15 minute limit exceeded)`
          )
        );
      };

      xhr.onerror = () => {
        if (options?.signal) {
          options.signal.removeEventListener("abort", abortHandler);
        }

        reject(
          new ApiError(
            0,
            `"${file.name}" network error during upload`
          )
        );
      };

      xhr.onabort = () => {
        if (options?.signal) {
          options.signal.removeEventListener("abort", abortHandler);
        }

        reject(new ApiError(0, `"${file.name}" upload cancelled`));
      };

      xhr.send(formData);
    });

    results.push(result);
  }

  // Ensure progress reaches 100% after all files finish.
  options?.onProgress?.({
    fileName: files.length === 1 ? files[0].name : `${files.length} files`,
    progress: 100,
  });

  return results;
}