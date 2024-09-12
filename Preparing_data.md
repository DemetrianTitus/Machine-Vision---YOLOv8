# Instructions on data preparation
## Step 1: Prerequisite

Ensure you have the following installed:
- **Python 3.9.13**: [Download Python](https://www.python.org/downloads/)

***Note:*** It is possible to use Python and create a virtual environment in VS Code to download images and labels directly within VS Code.

## Step 2: Download and Set Up `OIDv4_ToolKit`

This project uses the `OIDv4_ToolKit` to download and visualize single or multiple classes from the Open Images v4 dataset. You can find more details about the toolkit and its usage in its [GitHub repository](https://github.com/EscVM/OIDv4_ToolKit).

---
After downloading the `OIDv4_ToolKit`, navigate to the directory on your PC. For example:

```bash
cd C:\Users\(User Name)\ ... \OIDv4_ToolKit
``` 

Install the necessary Python libraries from the `requirements.txt` file by running:

  ```bash 
  pip install -r requirements.txt
  ```

## Step 3: Install Custom Classes and Define Constraints

Let's assume you want to download three specific classes:
1. Car
2. Airplane
3. Traffic light

To download these classes, run the following command inside the terminal:

```bash
python main.py downloader --classes Car Airplane Traffic_light --type_csv train --limit 2000
```

- In the example above, the command will download a maximum of `2000` images for each of the three classes.

***Note:*** If there are fewer than `2000` images available for a specific class, the command will download all available images for that class, which may be less than the specified limit.

---

### Example with Additional Constraints
- Here is another example using different constraints:

```bash
python main.py downloader --classes Car Bus Traffic_light Traffic_sign Street_light Person --type_csv train --sub h --image_IsOccluded 0 --image_IsTruncated 0 --image_IsGroupOf 0 --image_IsDepiction 0 --image_IsInside 0 --n_threads 20 --limit 8000
```

***Note:*** For classes with multiple words like `Traffic light`, use an underscore to connect them in the command, e.g., `Traffic_light`.

---

### Additional Information

- For more details about other constraints and features, you can check out the [OIDv4_ToolKit GitHub repository](https://github.com/EscVM/OIDv4_ToolKit.git) by EscVM.


## Step 4: Adding Classes

To specify the classes you want to download, follow these steps:

1. Open the `classes` folder inside the `OIDv4_ToolKit` directory.
2. Enter the desired classes, with each class on a separate line.

***Note:*** For classes with multiple words, such as `Traffic light`, write them with a space, like `Traffic light`.

---

#### Example of the contents of the `classes.txt` file:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div>
        <pre>
Car
Airplane
Traffic light
...
...
...
        </pre>
    </div>
</body>
</html>


## Step 5: Verify `.txt` Files

- Open each `.txt` file inside the `Label` folder and check if the first word in the `.txt` file corresponds to `Car` (for the Car folder) or `Airplane` (for the Airplane folder), as shown in the example below.

### Folder OID currently looks like:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div>
        <pre>
OID
├── csv_folder
│   ├── class-descriptions-boxable.csv
│   └── train-annotations-bbox.csv
├── Dataset
│   └── train
│       ├── Car
│       │   ├── Label
│       │   │   ├── 000f31e71b56641e.txt        <- Example of .txt file contents ->             
│       │   │   ├── 000f234939e98c68.txt            
│       │   │   └── ...                
│       │   ├── 000f31e71b56641e.jpg              
│       │   ├── 000f234939e98c68.jpg
│       │   └── ...
│       ├── Airplane
│       │   ├── Label
│       │   │   ├── 00a587aee58e37fe.txt        <- Example of .txt file contents ->             
│       │   │   ├── 00bb028447c9c791.txt             
│       │   │   └── ...                
│       │   ├── 00a587aee58e37fe.jpg
│       │   ├── 00bb028447c9c791.jpg
│       │   └── ...
        </pre>
    </div>
</body>
</html>

### Example of .txt file contents

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="box-container">
        <div class="box">
            <pre>
| Object    | X      | Y          | Width | Height             |
|-----------|--------|------------|-------|--------------------|
| Car       | 522.24 | 185.692829 | 601.6 | 253.56632700000003 |
| Airplane  | 100.24 | 50.692829  | 200.6 | 150.56632700000003 |
            </pre>
        </div>
    </div>
</body>
</html>


## Step 6: (Optional) Safety Measure

- **Compress** the `OID` folder into a .zip file using compression level 0. This precaution helps prevent the need for re-downloading images and labels in the event that the label conversion process encounters issues in the subsequent steps.


## Step 7: Convert Classes to `yolo` Format

Run the following command to `convert annotations` to the `yolo` format:

```bash
python convert_annotations.py
```

### Current structure of the OID folder:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="box-container">
        <div class="box">
            <pre>
OID
├── csv_folder
│   ├── class-descriptions-boxable.csv
│   └── train-annotations-bbox.csv
├── Dataset
│   └── train
│       ├── Car
│       │   ├── Label
│       │   │   ├── 000f31e71b56641e.txt        <- Example of .txt file contents ->             
│       │   │   ├── 000f234939e98c68.txt            
│       │   │   └── ...                
│       │   ├── 000f31e71b56641e.jpg              
│       │   ├── 000f234939e98c68.jpg
│       │   └── ...
│       ├── Airplane
│       │   ├── Label
│       │   │   ├── 00a587aee58e37fe.txt        <- Example of .txt file contents ->             
│       │   │   ├── 00bb028447c9c791.txt             
│       │   │   └── ...                
│       │   ├── 00a587aee58e37fe.jpg
│       │   ├── 00bb028447c9c791.jpg
│       │   └── ...
            </pre>
        </div>
    </div>
</body>
</html>

### Example of .txt file contents
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div class="box-container">
        <div class="box">
            <pre>
| ID  | X      | Y          | Width | Height              |
|-----|--------|------------|-------|---------------------|
| 0   | 522.24 | 185.692829 | 601.6 | 253.56632700000003  |
| 1   | 100.24 | 50.692829  | 200.6 | 150.56632700000003  |
            </pre>
        </div>
    </div>
</body>
</html>


- Verify for each class whether the "converted .txt files" start with numbers instead of class names.

- If the order in `classes.txt` specifies Car, Airplane, then when converting the `.txt` files, the class name "Car" should be replaced with the number "0", and "Airplane" should be replaced with the number "1".


## Step 8: Remove `Label` Folder

- Delete the `Label` folder within each class directory.


### The structure of OID after deletion is as follows:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div>
        <pre>
OID
├── csv_folder
│   ├── class-descriptions-boxable.csv
│   └── train-annotations-bbox.csv
├── Dataset
│   └── train
│       ├── Car
│       │   ├── 000f31e71b56641e.jpg
│       │   ├── 000f31e71b56641e.txt        <- converted labels .txt
│       │   ├── 000f234939e98c68.jpg
│       │   ├── 000f234939e98c68.txt        <- converted labels .txt
│       │   └── ...
│       ├── Airplane
│       │   ├── 00a587aee58e37fe.jpg
│       │   ├── 00a587aee58e37fe.txt        <- converted labels .txt
│       │   ├── 00bb028447c9c791.jpg
│       │   ├── 00bb028447c9c791.txt        <- converted labels .txt
│       │   └── ...
            </pre>
        </div>
    </div>
</body>
</html>



## Step 9: (Suggestion) Adding Prefix to Image/Text File Names

1. Open `Windows PowerShell` and navigate to the directory containing the images and text files to which you wish to add a prefix.

    ```powershell
    (Get-ChildItem -File) | Rename-Item -NewName {$_.Name -replace "^","PREFIX"}
    ```

2. Note: Replace `"PREFIX"` with the desired prefix, for example:

    - `"Apple_"`
    - `"Banana_"`
    - `"Car_"`
    - `"Airplane_"`

 ### The current structure of OID is as follows:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div>
        <pre>
OID
├── csv_folder
│   ├── class-descriptions-boxable.csv
│   └── train-annotations-bbox.csv
├── Dataset
│   └── train
│       ├── Car
│       │   ├── Car_000f31e71b56641e.jpg
│       │   ├── Car_000f31e71b56641e.txt
│       │   ├── Car_000f234939e98c68.jpg
│       │   ├── Car_000f234939e98c68.txt
│       │   └── ...
│       ├── Airplane
│       │   ├── Airplane_00a587aee58e37fe.jpg
│       │   ├── Airplane_00a587aee58e37fe.txt
│       │   ├── Airplane_00bb028447c9c791.jpg
│       │   ├── Airplane_00bb028447c9c791.txt
│       │   └── ...
            </pre>
        </div>
    </div>
</body>
</html>

## Step 10: Proper Distribution for YOLO Training

***Note:*** Pay attention to the capitalization of folder names. Use an approximate ratio of 70:30 for training and validation data.

1. Within the OID folder, in each class, right-click and select "Sort by Type." Then, take approximately 30% of the most recent `.txt` and `.jpg` files and move them to the `val` folder as shown below. The remaining ~70% of files should be placed in the `train` folder.

2. The number of selected files can be viewed in the bottom left corner of the `File Explorer` window in Windows.

3. It is crucial that each `.jpg` file has its corresponding `.txt` file in the appropriate folder (either `train` or `val`). Refer to the image below for clarification.

4. (Optional) Create a separate file named `Number_of_Data.txt` to record the number of train and validation files. This is useful for writing reports.

### Example of Distribution of Train and Validation Data:

<div align="center">

| Classes       | Train | Validation |
|---------------|-------|------------|
| Ball          | 2500  | 980        |
| Book          | 5000  | 2894       |
| Bottle        | 5000  | 4994       |
| Doll          | 2000  | 1383       |
| Headphones    | 800   | 264        |
| Mug           | 1100  | 498        |
| Teddy bear    | 800   | 220        |
| **Total**     | **17,200** | **11,223** |

</div>

### Final Structure of the "Dataset" Folder:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <div>
        <pre>
Dataset
├── train
│   ├── images
│   │   ├── Airplane_00a587aee58e37fe.jpg
│   │   ├── Airplane_00bb028447c9c791.jpg
│   │   ├── Car_000f31e71b56641e.jpg
│   │   ├── Car_000f234939e98c68.jpg
│   │   └── ...
│   └── labels
│       ├── Airplane_00a587aee58e37fe.txt
│       ├── Airplane_00bb028447c9c791.txt
│       ├── Car_000f31e71b56641e.txt
│       ├── Car_000f234939e98c68.txt
│       └── ...
├── val
│   ├── images
│   │   ├── Airplane_00a435aee58e37fe.jpg
│   │   ├── Airplane_44bb028447c9c791.jpg
│   │   ├── Car_123f31e71b56641e.jpg
│   │   ├── Car_123f234939e98c68.jpg
│   │   └── ...
│   └── labels
│       ├── Airplane_00a435aee58e37fe.txt
│       ├── Airplane_44bb028447c9c791.txt
│       ├── Car_123f31e71b56641e.txt
│       ├── Car_123f234939e98c68.txt
│       └── ...
            </pre>
        </div>
    </div>
</body>
</html>
