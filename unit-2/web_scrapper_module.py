import requests 

# Define our class
class Get_Github_File:

    def __init__(self, url, output_file):
        """
        Init function
        """
        self.url = url
        self.output_file = output_file

    def get_file(self):
        """
        Download file
        """

        response = requests.get(self.url)

        with open(self.output_file, "w", encoding="utf-8") as file:
            file.write(response.text)

        print(f"File successfully downloade as '{self.output_file}'")

