import os
from pathlib import Path


class ModelCheckpoint(object):
    """ ModelCheckpoint handler can be used to periodically save objects to disk.

    Args:
        dirname (str):
            Directory path where objects will be saved.
        save_fn (callable):
            Function that will be called to save the object. It should have the signature `save_fn(obj, path)`.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be removed.
        gcs_bucket (str, optional):
            If provided, will sync the saved model to the specified GCS bucket.
    """

    def __init__(self, dirname, save_fn, n_saved=1, gcs_bucket=None):
        self._dirname = Path(dirname).expanduser()
        self._n_saved = n_saved
        self._save_fn = save_fn
        if gcs_bucket.startswith("gs://"):
            gcs_bucket = gcs_bucket[5:]
        self._gcs_bucket = gcs_bucket
        self._saved = []

    def _check_dir(self):
        self._dirname.mkdir(parents=True, exist_ok=True)

        # Ensure that dirname exists
        if not self._dirname.exists():
            raise ValueError(
                "Directory path '{}' is not found".format(self._dirname))

    def save(self, obj, name, sync_gcs=True):
        self._check_dir()
        path = self._dirname / name
        self._save_fn(obj, str(path))
        self._saved.append(path)
        print(f"Saved model to {path}")

        if self._gcs_bucket is not None and sync_gcs:
            fname = "latest" + path.suffix
            gcs_url = Path(self._gcs_bucket) / fname
            gcs_url = f"gs://{gcs_url}"
            os.system(f"gsutil cp {path} {gcs_url} >> gcs_sync.log 2>&1 &")
            print("Sync to GCS: ", gcs_url)

        if len(self._saved) > self._n_saved:
            path = self._saved.pop(0)
            os.remove(path)
