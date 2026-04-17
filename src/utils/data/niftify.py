class Niftify:

    def __init__(self, zip_path, extract_to, nifti_path):
        self.zip_path = zip_path
        self.extract_to = extract_to
        self.nifti_path = nifti_path
        
    def unzip_file(self):
        """
        Unzips a ZIP file to the specified directory.
    
        :param zip_path: Path to the .zip file
        :param extract_to: Directory where files will be extracted
        """
        try:
            if not os.path.isfile(self.zip_path):
                raise FileNotFoundError(f"ZIP file not found: {self.zip_path}")
    
            os.makedirs(self.extract_to, exist_ok=True)
    
            # Open and extract the ZIP file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_to)
                print(f"Extracted {len(zip_ref.namelist())} files to '{self.extract_to}'")
    
        except zipfile.BadZipFile:
            print("Error: The file is not a valid ZIP archive.")
        except PermissionError:
            print("Error: Permission denied while accessing files.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def to_nifti(self):
        assert os.path.isdir(self.extract_to), f"{self.extract_to} is not a directory"
        
        os.makedirs(self.nifti_path, exist_ok=True)
        dicom2nifti.convert_directory(self.extract_to, self.nifti_path)

    def run(self):
        self.unzip_file()
        self.to_nifti()