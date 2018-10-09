import youtube_dl
import re
import webvtt

max_len_of_sent = 40

# credit: pdemange. From his collector.py
class CaptionCollector:

    def __init__(self, lang='en'):
        self.language = lang
        self.ydlOpts = {'writesubtitles': lang,
                        'skip_download': True,
                        'outtmpl':'subtitles.vtt'}
        self.urlFinder = re.compile('(https?://\S+)')

    def downloadSubs(self, video, filename='subtitles'):
        self.ydlOpts['outtmpl'] = filename+'.vtt'
        with youtube_dl.YoutubeDL(self.ydlOpts) as ydl:
            ydl.download([video])

    def readAllCaptions(self, file):
        captionsList = []
        for caption in webvtt.read(file):
            captionsList.append(caption.text)
        return captionsList

    def formatCaptions(self, captions, replacementDict=None):
        if replacementDict is not None:
            newCaptions = []
            if isinstance(captions, list):
                for caption in captions:
                    if isinstance(replacementDict, dict):
                        for substring, replacement in replacementDict.items():
                            print('Replacing %s with %s' %(substring,replacement))
                            caption = caption.replace(substring, replacement)
                        newCaptions.append(caption)
                    else:
                        print('Replacement dictionary is not in the right format!')
                        break
            return newCaptions
        else:
            print('Nothing to format!')

    def downloadFromList(self, file, subtitleFileName='subtitles'):
        urls = None
        with open(file, 'r') as f:
            urls = self.urlFinder.findall(f.read())
            f.close()

        for i, url in enumerate(urls):
            temp = subtitleFileName+'_'+str(i)
            self.downloadSubs(url, temp)

class directory_of_letters:

    def __init__(self, list):
        self.dictionary = dict()
        for i in range(len(list)):
            self.dictionary[i] = list[i]

if __name__ == '__main__':
    captions_class = CaptionCollector()
    # captions_class.downloadSubs("https://youtu.be/otwkRq_KnG0", "8-bitryan")
    caption = captions_class.readAllCaptions('8-bitryan.en.vtt')
    sent = str.join(' ', caption)
    ordered_list = sorted(list(set(sent.lower())))
    dict_of_letters = directory_of_letters(ordered_list)
    print(dict_of_letters.dictionary)
    print(dict_of_letters.dictionary[1])
    # print(sent)
