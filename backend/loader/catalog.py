import os
import json
import hashlib
import uuid
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from logger import logging
from exception import MedicalAssistantException


class DocumentCatalog:
    def __init__(self, catalog_path: str = "upload/catalog.json"):
        self.catalog_file = Path(catalog_path).resolve()
        self.catalog_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.catalog_file.exists():
            with open(self.catalog_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_catalog(self) -> Dict[str, Any]:
        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_catalog(self, catalog: Dict[str, Any]):
        with open(self.catalog_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    def compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_catalog(self) -> List[Dict[str, Any]]:
        catalog = self._load_catalog()
        return list(catalog.values())

    def register_document(
        self,
        file_path: Path,
        filename: str,
        category: str = "general",
        source_url: Optional[str] = None,
        chunks_count: int = 0
    ) -> Tuple[str, bool]:
        """
        Registers a document in the catalog.
        Returns: (doc_id: str, is_new: bool)
        """
        try:
            file_hash = self.compute_file_hash(file_path)
            catalog = self._load_catalog()

            # Check if document with same content hash already exists
            for doc_id, meta in catalog.items():
                if meta.get("file_hash") == file_hash:
                    logging.info(f"Document '{filename}' already exists in catalog with doc_id '{doc_id}'")
                    return doc_id, False

            new_doc_id = str(uuid.uuid4())
            doc_meta = {
                "doc_id": new_doc_id,
                "filename": filename,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "category": category.lower(),
                "source_url": source_url or str(file_path),
                "chunks_count": chunks_count,
                "created_at": os.path.getctime(file_path) if file_path.exists() else 0.0
            }
            catalog[new_doc_id] = doc_meta
            self._save_catalog(catalog)
            logging.info(f"Registered new document in catalog: '{filename}' (doc_id: {new_doc_id})")
            return new_doc_id, True

        except Exception as e:
            logging.error(f"Error registering document in catalog: {e}")
            raise MedicalAssistantException(e, sys)

    def download_url(self, url: str, dest_dir: Path) -> Path:
        """
        Downloads a document from a remote URL into destination directory.
        """
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            filename = os.path.basename(url.split("?")[0])
            if not filename or not filename.endswith((".pdf", ".txt", ".csv", ".docx")):
                filename = f"remote_doc_{uuid.uuid4().hex[:8]}.pdf"

            dest_path = dest_dir / filename
            logging.info(f"Downloading remote document from '{url}' to '{dest_path}'...")
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())

            logging.info(f"Successfully downloaded remote document to '{dest_path}'")
            return dest_path
        except Exception as e:
            logging.error(f"Error downloading document from URL '{url}': {e}")
            raise MedicalAssistantException(e, sys)

    def remove_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            catalog = self._load_catalog()
            if doc_id in catalog:
                removed_meta = catalog.pop(doc_id)
                self._save_catalog(catalog)
                # Remove file if local
                file_path = Path(removed_meta["file_path"])
                if file_path.exists():
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logging.warning(f"Could not remove local file {file_path}: {e}")
                logging.info(f"Removed document doc_id '{doc_id}' from catalog.")
                return removed_meta
            return None
        except Exception as e:
            logging.error(f"Error removing document from catalog: {e}")
            raise MedicalAssistantException(e, sys)
