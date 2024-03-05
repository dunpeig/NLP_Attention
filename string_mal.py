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

