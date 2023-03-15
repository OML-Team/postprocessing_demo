# Postprocessing demo

In order to run the demo one needs to execute the following steps:

1. Run `csv_generator.py` to generate dataframes for query and gallery sets.
2. Run `prepare_images_zips.py` to generate dataframes for query and gallery that
only contains queries for which there were some changes during the postprocessing step,
and also zip-file with all the query and gallery images.
3. Run `streamlit main.py` in order to run the demo.
