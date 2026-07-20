# datinghelper

to create a profile and pictures from finya profile id

`py finya_fetch.py <id>`

after profile & pictures created, to analyze pictures and add the gathered information to `profile.xml`

`py analyze_pictures.py <folder name inside profiles>`

example:

```
py finya_fetch.py some_id
py analyze_pictures.py finya_sample_username
py icebreaker.py finya_my_username finya_sample_username
```