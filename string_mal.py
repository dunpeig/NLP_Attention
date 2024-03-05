# Pythin String manipulation 
## reduce empty space or strings
input_str = '  good weather today  , very sunny  '
print(input_str.strip())
print(input_str.rstrip())
print(input_str.lstrip())

input_str = 'AAAgood weather today  , very sunnyAAA'
print(input_str.strip('A'))
print(input_str.lstrip('A'))
print(input_str.rstrip('A'))
# replace
input_str = '  good weather today  , very sunny  '
print(input_str.replace('today','yesterday'))
print(input_str.replace('today',''))
# find 
print(input_str.find('today'))

# assess
print(input_str.isalpha())
input_str = 'abc'
print(input_str.isalpha())
print(input_str.isdigit())
input_str = '123'
print(input_str.isdigit())

# divide and merge
input_str = '  good weather today  , very sunny  '
input_str = input_str.split(' ')
print(input_str)
print(' '.join(input_str))



# Re 

input = 'natural language processing. 12abc789'
import re 
pattern = re.compile(r'.')
print(re.findall(pattern, input))

pattern = re.compile(r'[abc]')
print(re.findall(pattern, input))

# a to z and A to Z
pattern = re.compile(r'[a-zA-Z]')
print(re.findall(pattern, input))
# not 
pattern = re.compile(r'[^a-zA-Z]')
print(re.findall(pattern, input))
# or condition
pattern = re.compile(r'[^a-zA-Z]|[0-9]')
print(re.findall(pattern, input))

# \d is number [0-9] , \D is [^\d] not number, 
pattern = re.compile(r'\d')
print(re.findall(pattern, input))

pattern = re.compile(r'\D')
print(re.findall(pattern, input))

# \w number and letter, \W not number or letter

pattern = re.compile(r'\w')
print(re.findall(pattern, input))

pattern = re.compile(r'\W')
print(re.findall(pattern, input))

# space 
pattern = re.compile(r'\s')
print(re.findall(pattern, input))

# repete 
input = 'natural language processing. 12ab3c789'

# zero or more 
pattern = re.compile(r'\d*')
print(re.findall(pattern, input))
#  1 or more 
pattern = re.compile(r'\d+')
print(re.findall(pattern, input))
# 0 or 1
pattern = re.compile(r'\d?')
print(re.findall(pattern, input))

# m times 
pattern = re.compile(r'\d{2}')
print(re.findall(pattern, input))

# m to n times 
pattern = re.compile(r'\d{1,2}')
print(re.findall(pattern, input))

# match and search 
input = '1natural language processing. 12ab3c789'
pattern = re.compile(r'\d')
# match from the begiining, if not macthed at position 0 then false 
match = re.match(pattern, input)
print(match.group())

# replace and modify 

pattern = re.compile(r'\d')
# substitute
print(re.sub(pattern, 'number',input))
# also gives how many times replacement happened 
print(re.subn(pattern, 'number',input))

# split 
input = 'natural123machinelearning3deeplearning'
pattern = re.compile(r'\d+')
print(re.split(pattern, input))

# naming 
pattern = re.compile(r'(?P<dota>\d+)(?P<lol>\D+)')
m =re.search(pattern, input)
print(m.group('dota'))
print(m.group('lol'))

# 
input = 'number 222-121-2123'
pattern = re.compile(r'(\d\d\d-\d\d\d-\d\d\d)')
m = re.search(pattern, input)
print(m.groups())
