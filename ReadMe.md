## Prerequisites 

- Virtual Environment with Python < 3.13 is needed, to make tensorflow work
- Always execute the main script from the root folder, aka `python3 src/main.py`!
- Start WebApp with:

```bash
streamlit run src/app.py
```

## Modules
Modules are the building blocks of the pipeline. Each Module should execute (in theory) exactly one task.
Currently following modules exist:

- **PdfConverter**: Converts PDF to jpg pages
    - Input
        - `output_folder: str`
    - Output: jpg images in `output_folder`

- **TableRotator**: Rotates jpg images
    - Input
        - `output_folder: str`
    - Output: rotated jpg images in `output_folder`

- **TatrExtractor**: Extract table structure from jpg images
    - Input
        - `output_folder: str`
    - Output: extracted table images in `output_folder`

- **ColumnExtractor**: Tries to extract all columns from table images
    - Input
        - `minFoundColumns: int`: Min amount of columns to be found to be an "eligible" page
        - `try_experimental_unify: bool`: Experimental feature. Tries to map columns based on their extracted width
    - Output `list[ColumnExtractorResult]`
        - `columns_rgb: list[np.ndarray]`: Cut-out rgb columns
        - `columns_gray: list[np.ndarray]`: Cut-out gray-scaled columns
        - `split_widths: list[float]`: Column widths

- **RowExtractor**: Tries to extract all rows inside a column
    - Output `list[RowExtractorResult]`
        - `columns: list[list[np.ndarray]]`: Cut-out gray cell for each column and row

- **CellDenoiser**: Denoises each cell
    - Output `list[CellDenoiserResult]`
        - `columns: list[list[np.ndarray]]`: Denoised gray cell for each column and row