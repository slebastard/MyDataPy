## MyDataPy
## Data processing shelf
## Simon Lebastard - Nov 2018

## External requirements ###########################

# Identification with Google account to access data
#from google.colab import auth
#auth.authenticate_user()
#
## This shelf requires gspread. To install:
## !pip install --upgrade -q gspread
#import gspread
#from oauth2client.client import GoogleCredentials
#
#gc = gspread.authorize(GoogleCredentials.get_application_default())

## Internal requirements ##########################


###################################################

#def dataFromSheet(workbook, sheetNumber=0):
#  """Reads data from a specific sheet in a Google workbook"""
#  worksheet = gc.open(workbook).get_worksheet(sheetNumber)
#  data = pd.DataFrame.from_records(worksheet.get_all_values()).values.astype(float)
#  return data