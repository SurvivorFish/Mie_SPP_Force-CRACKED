file1 = 'balovstvo/scattering1.txt'
file2 = 'balovstvo/scattering1-crop.txt'

lines = []
with open(file1, 'r') as f:
    for i in range(22):
        f.readline()
    read = f.readline()
    while read != '' and read != ' ':
        lines.append(read)
        read = f.readline()


with open(file2, 'w') as f:
    ans = ''
    for i in range(len(lines)):
        line = lines[i].split(', ')
        ans = ans + line[0] + ', ' + line[1] + '\n'
    f.write(ans)
