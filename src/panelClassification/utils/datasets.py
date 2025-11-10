# ---- Add this new builder next to your existing build_image_datasets ----
import os, glob, math, random
import tensorflow as tf

def build_image_datasets_stratified(
    conf,
    *,
    subset: str = "both",      # "both" | "train" | "val"
    shuffle: bool = True,
    use_aug: bool = False,
) -> tuple[tf.data.Dataset|None, tf.data.Dataset|None, list[str]]:
    """
    Returns (train_ds, valid_ds, class_names) with a **stratified** split,
    so every class appears in both train and val (when possible).
    """
    img_h, img_w = conf.params_image_size[:2]
    batch_size   = conf.params_batch_size
    data_dir     = str(conf.training_data)
    val_split    = float(getattr(conf, "val_split", 0.20))
    seed         = int(getattr(conf, "seed", 42))
    rng          = random.Random(seed)

    # --- single high-quality resizer  ---
    resizer = tf.keras.layers.Resizing(img_h, img_w, interpolation="bicubic", antialias=True)

    # ---------- 1) scan files & labels ----------
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    per_class_files = []
    for cls in class_names:
        folder = os.path.join(data_dir, cls)
        files = []
        for f in glob.glob(os.path.join(folder, "*")):
            ext = f.lower().rsplit(".", 1)[-1]
            if ext in {"jpg","jpeg","png","bmp","tif","tiff"}:
                files.append(f)
        files.sort()
        per_class_files.append(files)

    # ---------- 2) stratified split (manual) ----------
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for ci, files in enumerate(per_class_files):
        if shuffle:
            rng.shuffle(files)
        n = len(files)
        # hold out roughly val_split per class
        n_val = max(1 if n>=2 else 0, int(round(n * val_split)))
        # if class has <2 items, you can't stratify -> put all in train
        val_part = files[:n_val]
        train_part = files[n_val:]
        # If val would be empty, push one sample there to avoid missing class
        if n>=2 and len(val_part)==0 and len(train_part)>0:
            val_part = [train_part.pop(0)]

        val_paths.extend(val_part);   val_labels.extend([ci]*len(val_part))
        train_paths.extend(train_part);train_labels.extend([ci]*len(train_part))

    # ---------- 3) TF datasets ----------
    def _load_and_resize(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)   # 0..255
        img = resizer(img)               # single, high-quality resize
        return img, label


    def _ds(paths, labels, do_shuffle):
        if not paths:
            return None
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if do_shuffle:
            ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # ---------- 4) choose preprocess_input by backbone ----------
    name = str(conf.base_model_name).lower()
    if name == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input as _ppi
    elif name == "efficientnet_b0":
        from tensorflow.keras.applications.efficientnet import preprocess_input as _ppi
    elif name == "efficientnet_v2_s":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as _ppi
    else:
        def _ppi(x): return tf.keras.layers.Rescaling(1./255)(x)

    # ---------- 5) optional augmentation (train only) ----------
    aug = None
    if use_aug:
        aug = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),  # work in [0,1]

            tf.keras.layers.RandomFlip("horizontal"),

            tf.keras.layers.RandomRotation(
                0.05,                         
                fill_mode="constant",
                fill_value=0.5,                 # neutral gray border
                interpolation="bilinear",
            ),
            tf.keras.layers.RandomZoom(
                0.05,                         
                fill_mode="constant",
                fill_value=0.5,
                interpolation="bilinear",
            ),
            tf.keras.layers.RandomTranslation(
                0.05, 0.05,
                fill_mode="constant",
                fill_value=0.5,
                interpolation="bilinear",
            ),


            # Safety: keep values in-range after aug
            tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0)),
            tf.keras.layers.Lambda(lambda x: x * 255.0),  # back to 0..255 for preprocess_input
        ])



    def _pp_train(x, y):
        x = tf.cast(x, tf.float32)
        if aug is not None:
            x = aug(x)
        x = _ppi(x)
        return x, y

    def _pp_val(x, y):
        x = tf.cast(x, tf.float32)
        x = _ppi(x)
        return x, y

    train_ds = valid_ds = None
    if subset in ("both","train"):
        train_ds = _ds(train_paths, train_labels, do_shuffle=shuffle)
        if train_ds is not None:
            train_ds = train_ds.map(_pp_train, num_parallel_calls=tf.data.AUTOTUNE)\
                               .prefetch(tf.data.AUTOTUNE)
    if subset in ("both","val"):
        valid_ds = _ds(val_paths, val_labels, do_shuffle=False)
        if valid_ds is not None:
            valid_ds = valid_ds.map(_pp_val, num_parallel_calls=tf.data.AUTOTUNE)\
                               .prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, class_names
