import imaplib
import email
import getpass
import youtube_dl
import webvtt
import re

class EmailCollector:

    def __init__(self, username, password, imapServer='imap.gmail.com', port=993):
        self.gmail = self.login(username,password,imapServer,port)

    def login(self, user,passwd,imapS,p):
        connection = imaplib.IMAP4_SSL(imapS,p)
        connection.login(user,passwd)
        return connection

    def getBody(self, msg):
        if msg.is_multipart():
            return self.getBody(msg.get_payload(0))
        else:
            return msg.get_payload(None, True)

    def selectInbox(self, box):
        self.gmail.select(box)

    def searchInbox(self, searchString):
        retType, data = self.gmail.search(None, searchString)
        if retType == 'OK':
            return data[0].split()
        else:
            return False

    def getEmailBody(self, id):
        retType, data = self.gmail.fetch(id, '(RFC822)')
        if retType == 'OK':
            rawMsg = email.message_from_bytes(data[0][1])
            return self.getBody(rawMsg).decode('utf-8')
        else:
            return False


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


class DataHandler:

    def __init__(self, sentences, maxLen=40):
        self.sentencesCom = ' '.join(sentences)
        self.charIndices, self.indicesChars, self.charLen = self.getUniqueChars(self.sentencesCom)
        self.maxLen = maxLen
        self.training, self.testing = self.prepareData()

    def getUniqueChars(self, sentencesString):
        chars = sorted(list(set(sentencesString.lower())))
        charIndices = dict((c,i) for i, c in enumerate(chars))
        indicesChar = dict((i,c) for i, c in enumerate(chars))
        return charIndices,indicesChar,len(chars)

    def prepareData(self, step=3):
        allSent = []
        nextChars = []
        for i in range(0,len(self.sentencesCom)-self.maxLen, step):
            allSent.append(self.sentencesCom[i:i+self.maxLen])
            nextChars.append(self.sentencesCom[i+self.maxLen])

        x = np.zeros((len(allSent), self.maxLen, self.charLen), dtype=np.bool)
        y = np.zeros((len(allSent), self.charLen), dtype=np.bool)
        for i, sent in enumerate(allSent):
            for c, char in enumerate(sent):
                x[i,t,self.charIndices[c]]=1
            y[i,self.charIndices[nextChars[i]]]=1

        return x,y



if __name__ == '__main__':
    y = CaptionCollector()
    caps = y.readAllCaptions('subtitles_0.en.vtt')
    nCaps = y.formatCaptions(caps, {'\n':'','D:':'\nD:', 'A:':'\nA:','Arin:': '\nA:', 'Dan:':'\nD:', '(Arin)':'\nA:', '(Danny)':'\nD:'})
    print(nCaps)
    c = DataHandler(nCaps)
    print(c.prepareData())
    #y.downloadFromList('youtube_list.txt')
    #y.downloadSubs('https://www.youtube.com/watch?v=xNnItWbMhL8', 'GG')


