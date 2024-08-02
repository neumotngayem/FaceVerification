# Face Verification

**Step 1**: Create models folder

**Step 2**: Download models following this link and put into models folder:
https://drive.google.com/drive/folders/1oOh45sI7HCWvlC7JzbpFcK5JQ81jg2Gc?usp=drive_link

# CLI

For registering the face:

```
python main.py -t r -p {image_folder_path} -n {name}
```

For verify the face:

```
python main.py -t v -p {image_path}
```

# API

Starting API server:

```
fastapi dev main_api.py
```
