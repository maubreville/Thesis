"""CLEdB.py: Database routines for CLE-classification database"""
__author__ = "Marc Aubreville"
__license__ = ""
__version__ = "0.0.1"


import sqlite3
import numpy as np
import os
##########################################################################
## The CLE database has the following structure:
##
##   CLEdatabases: Each database consists of a number of sequences (movies),
##                 and may be from a different hospital or anatomical region
##                 It has the following fields:
##                       id:          Unique ID
##                       path:        Path to the sequence files (mkt files)
##                       description: Description of the database
##                       authors:     Medical authors of the dataset
##   CLEsequences: Each sequence consists of a number of images (frames),
##                 It has the following main fields:
##                       id:         Unique ID
##                       patientID:  Unique identifier for patient (numerical)
##                       patient:    Patient identifier (string)
##                       fileId:     Unique identifier for a file (a file may contain in multiple CLE sequences)
##                       file:       MKT file name
##                       subfolder:  Subfolder containing MKT files
##                       database:   Link to CLEdatabases.id
##
##   CLEframes:    Single images of the CLE database.
##                 It has the following main fields:
##                       id:            Unique ID
##                       sequenceID:    Link to CLEsequences.id
##                       frameIdx:      Frame index in the MKT file (0: first image in file)
##                       cellStructure: Cell classification (-1: unknown, 0: normal
##                                      epithel, 1: carcinoma, 2: dysplasia)
##                       anatomicalLocation: Location where the sample was taken
##                                           (0: upper alveolar ridge, 1: lower inner
##                                            labium, 2: palatal region, 3: lesion region,
##                                            4: vocal folds)
##                       gaussianNoiseClass: Noisyness of the image (0: not, 10: completely)
##                       motionArtifactClass: Motion artifacts in image (0: none, 10: only artifacts)
##                       illuminationArtifactClass: Illumination artifacts in the image (deprecated)
##                       imageQuality:  Subjective quality of image (0: bad, 1: neutral, 2: good)
##
##   CLEregions:   Regions of interest / region annotations within a single frame.
##                 Fields:
##                       id:            Unique ID
##                       frameId:        Link to CLEframes.id
##                       regionType:     Type of annotation (0: motion artifact, 1: noise artifact, 2: other artifact,
##                                       3: Other ROI/no artifact, e.g. anatomical structures)
##                       x1,x2,y1,y2:    Coordinates within image [x1:x2,y1:y2]
##
##########################################################################



##########################################################################
## CLEdBSequence: Class for CLE sequences in database
##########################################################################

class fileBatch:
    filename = ''
    frames = []

class CLEdBSequence:
    # Define main class fields
    idx = 0
    patient = ''
    file = ''
    fromIndex = 0
    toIndex = 0
    DBidx = 0
    ukeId = 0
    fps=0
    lengthDetected = 0
    length = 0
    filePath = ''
    patientUID = 0
    patientId = 0
    subfolder = ''

    ##########################################################################
    ## loadSequenceEntry: Load a database sequence  (renamed from loadEntry)
    ##########################################################################
    def loadSequenceEntry(self,idx=0):
      DB = CLEdB();
      self.DBidx=idx;
      DB.c.execute('SELECT file, patient, fromIndex, toIndex,ukeLength,fps,ukeId,path,subfolder FROM CLEsequences LEFT JOIN CLEdatabases on CLEdatabases.id == CLEsequences.database WHERE id='+str(idx));
      result=DB.c.fetchone()
      self.file = result[0]
      self.patient = result[1]
      self.fromIndex = int(result[2])
      self.toIndex = int(result[3])
      self.length = result[4]
      self.fps=result[5]
      self.ukeId=result[6]
      self.filePath = result[7]
      self.subfolder = result[8]


##########################################################################
## CLEregion: ROI class for CLEdB
##########################################################################

class CLEregion:
    frameId = 0
    regionType = -1
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0


##########################################################################
## CLEframe: Class for CLE frames (i.e. single pictures)
##########################################################################

class CLEframe:
    cellStructure=0
    imageQuality=1
    anatomicalLocation=-1;
    gaussianNoiseClass=-1;
    illuminationArtifactClass=-1;
    motionArtifactClass=-1;
    motionArtifactLabel=-1

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

#def convert_array(text):
#    out = io.BytesIO(text)
#    out.seek(0)
#    return np.load(out)

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    print(len(text))
    return  np.frombuffer(text, np.float32)#np.load(out)


##########################################################################
## CLEdB: Main class
##########################################################################

class CLEdB:

    ##########################################################################
    ## Constructur. Expects SQLite DB "CLEdb.db" in current directory, unless
    ## stated otherwise.
    ##########################################################################

    def __enter__(self):
        pass

    def __init__(self, dbpath='CLEdB.db'):


        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        if os.path.isfile(dbpath):
            self.conn = sqlite3.connect(dbpath, detect_types=sqlite3.PARSE_DECLTYPES)
        else:
            raise Exception('DB file not found. Please check path.')

        self.c=self.conn.cursor()

    ##########################################################################
    ## numSequences: Returns total number of CLE sequences
    ##########################################################################

    def numSequences(self):
        self.c.execute('SELECT COUNT(*) FROM CLEsequences');
        return self.c.fetchone()[0]

    ##########################################################################
    ## filesForDatabase: Returns all files contained in a database
    ##########################################################################

    def filesForDatabase(self, db):
        self.c.execute('SELECT fileId FROM CLEsequences WHERE database == '+str(db)+ ' group by fileId');
        filelist=list()
        for file in self.c.fetchall():
            filelist.append(file[0])
        return filelist

    ##########################################################################
    ## numFiles: Returns the total number of files
    ##########################################################################

    def numFiles(self):
        self.c.execute('SELECT fileId FROM CLEsequences group by fileId');
        return len(self.c.fetchall())

    ##########################################################################
    ## sequencesForFile: Find all CLE sequences for a certain file ID
    ##########################################################################

    def sequencesForFile(self, fileIdx):
        self.c.execute('SELECT id FROM CLEsequences WHERE fileId=='+str(fileIdx));
        return self.c.fetchall()

    ##########################################################################
    ## readRegions: Find all annotated regions for a frame
    ##########################################################################

    def readRegions(self, frameIdx):
        self.c.execute('SELECT regionType,x1,x2,y1,y2 FROM CLEregions WHERE' +
                       ' frameId=='+str(frameIdx))
        result = self.c.fetchall()
        return result

    ##########################################################################
    ## convertFrameIntoLocation: Converts frame numbers into locations
    ##########################################################################

    def convertFrameIntoLocation(self, frameIdxs):
        # Converts frame numbers into locations
        self.c.execute('SELECT anatomicalLocation, id from CLEframes')
        result = np.asarray(self.c.fetchall())
        idxs=np.searchsorted(result[:,1],frameIdxs)
        return result[idxs,0]

    ##########################################################################
    ## get a mask of the image (usually a filled circle), but removing
    ## all annotated artifacts, i.e. only valid image parts remain
    ##########################################################################

    def getMaskWithArtifacts(self,inMask, frameUID, regionClass = None):
        regions = self.readRegions(frameUID)
        mask = np.copy(inMask)
        for region in regions:
            if (region[0] < 4) and ((regionClass is None) or (region[0] ==regionClass)):  # 4 is ROI
                mask[region[3]:region[4], region[1]:region[2]] = 0
            
        return mask

    ##########################################################################
    ## getFrameRange:
    ## get minimum and maximum frame unique ID
    ##########################################################################

    def getFrameRange(self):
        self.c.execute('SELECT MIN(id), MAX(id) FROM CLEframes')
        result=self.c.fetchone()
        return result

    ##########################################################################
    ## getSequenceAndFrameIdxForFrame:
    ## Get sequence ID and frame IDX in file for a certain frame
    ##########################################################################

    def getSequenceAndFrameIdxForFrame(self,idx):
        self.c.execute('SELECT sequenceID,CLEframes.frameIdx-CLEsequences.fromIndex FROM CLEframes LEFT JOIN CLEsequences on CLEsequences.id == CLEframes.sequenceID WHERE CLEframes.id='+str(idx))
        result=self.c.fetchone();
        return result

    ##########################################################################
    ## readFrameEntry:
    ## Read details for a frame (as CLEframe class)
    ##########################################################################

    def readFrameEntry(self, frameUID):
        query='SELECT cellStructure, imageQuality, anatomicalLocation, gaussianNoiseClass, illuminationArtifactClass, motionArtifactClass, anatomicalLocation, frameIdx, stripeArtifact FROM CLEframes WHERE id='+str(frameUID)
        self.c.execute(query);
        result=self.c.fetchone();
        if (result is None):
           # entry has to be created, wasn't there before.
           self.c.execute('INSERT into CLEframes (sequenceID, frameIdx) VALUES ('+str(movieIdx)+','+str(frameIdx)+')');
           self.conn.commit();
           query='SELECT cellStructure, imageQuality, anatomicalLocation, gaussianNoiseClass, illuminationArtifactClass, motionArtifactClass,frameIdx FROM CLEframes WHERE id='+str(frameUID)
           self.c.execute(query);
           result=self.c.fetchone();

        retval = CLEframe()
        retval.cellStructure=int(result[0])
        retval.imageQuality=int(result[1])
        retval.anatomicalLocation=int(result[2])
        retval.gaussianNoiseClass=int(result[3])
        retval.illuminationArtifactClass=int(result[4])
        retval.motionArtifactClass=int(result[5])
        retval.anatomicalLocation=int(result[6])
        retval.frameIdx=int(result[7])
        retval.motionArtifactLabel=int(result[8])
        return retval


    ##########################################################################
    ## setFrameEntry:
    ## Set data fields for a CLEframe
    ## Usage example: db.setFrameEntry(frameUID=12345, imageQuality=2)
    ##########################################################################

    def setFrameEntry(self, frameUID, cellStructure=None, imageQuality=None, gaussianNoiseClass=None, motionArtifactLabel=None, illuminationArtifactClass=None, motionArtifactClass=None, anatomicalLocation=None):
        query = 'UPDATE CLEframes SET '
        if (illuminationArtifactClass is not None):
            query += 'illuminationArtifactClass=' + str(illuminationArtifactClass)+', '
        if (motionArtifactLabel is not None):
            query += 'stripeArtifact=' + str(motionArtifactLabel)+', '
        if (gaussianNoiseClass is not None):
            query += 'gaussianNoiseClass=' + str(gaussianNoiseClass)+', '
        if (motionArtifactClass is not None):
            query += 'motionArtifactClass=' + str(motionArtifactClass)+', '
        if (anatomicalLocation is not None):
            query += 'anatomicalLocation=' + str(anatomicalLocation)+', '
        if (cellStructure is not None):
            query += 'cellStructure=' + str(cellStructure)+', '
        if (imageQuality is not None):
            query += 'imageQuality=' + str(imageQuality)+', '
        query=query[:-2] # remove last 2 characters
        query += ' WHERE id='+str(frameUID)
        self.c.execute(query)
        self.conn.commit();

    ##########################################################################
    ## getAllFrameEntries:
    ## Get all entries for a certain CLE sequence
    ##########################################################################

    def getAllFrameEntries(self, sequenceID=0, constraint='') -> CLEdBSequence:
        entry = CLEdBSequence()
        entry.DBidx=sequenceID;

        if (len(constraint)>0):
            query = ('SELECT CLEsequences.file, CLEsequences.patient, CLEsequences.'
                     'fromIndex, CLEsequences.toIndex, CLEsequences.ukeLe'
                     'ngth,CLEsequences.fps,CLEsequences.ukeId , COUNT(*), CLEsequences'
                     '.patientId,CLEdatabases.path, CLEsequences.subfolder, CLEsequences.patientUID FROM '
                     'CLEsequences LEFT JOIN CLEframes on CLEsequences.id == '
                     'CLEframes.sequenceID LEFT JOIN CLEdatabases on CLEsequences.database == CLEdatabases.id '+constraint+' AND CLEsequences.id '
                     '== '+str(sequenceID)+' group by CLEframes.sequenceID')
        else:
            query = ('SELECT CLEsequences.file, CLEsequences.patient, CLEsequences.from'
                     'Index, CLEsequences.toIndex, CLEsequences.ukeLength,'
                     'CLEsequences.fps,CLEsequences.ukeId , COUNT(*), CLEsequences.'
                     'patientId,CLEdatabases.path, CLEsequences.subfolder, CLEsequences.patientUID FROM CLEsequences'
                     ' LEFT JOIN CLEframes on CLEsequences.id == CLEframes.'
                     'sequenceID LEFT JOIN CLEdatabases on CLEsequences.database == CLEdatabases.id WHERE CLEsequences.id == '+str(sequenceID)+' group by '
                     'CLEframes.sequenceID')
        print(query)
        self.c.execute(query)
        result=self.c.fetchone()
        if ((result is not None) and (len(result)>0)):
            entry.file = result[0]
            entry.patient = result[1]
            entry.fromIndex = int(result[2])
            entry.toIndex = int(result[3])
            print('nFrames: '+str(result))
            entry.length = result[4]
            entry.fps=result[5]
            entry.count=result[7]
            entry.patientId=result[8]
            entry.filePath=result[9]
            entry.subfolder=result[10]
            entry.patientUID=result[11]
            if (entry.subfolder is None):
                entry.subfolder=''
        else:
            entry.count = 0

        if (len(constraint)>0):
            print('SELECT CLEframes.id, CLEframes.frameIdx FROM CLEframes LEFT JOIN CLEsequences on CLEframes.sequenceID == CLEsequences.id '+constraint+' AND CLEframes.sequenceID == '+str(sequenceID))
            self.c.execute('SELECT CLEframes.id, CLEframes.frameIdx FROM CLEframes LEFT JOIN CLEsequences on CLEframes.sequenceID == CLEsequences.id '+constraint+' AND CLEframes.sequenceID == '+str(sequenceID))
        else:
            self.c.execute('SELECT CLEframes.id, CLEframes.frameIdx FROM CLEframes WHERE CLEframes.sequenceID == '+str(sequenceID));
        entry.frames = self.c.fetchall();

        return entry

    ##########################################################################
    ## getSequenceList:
    ## Get all sequences fulfilling a certain constraint
    ##########################################################################

    def getSequenceList(self, constraint):
        self.c.execute('SELECT CLEsequences.id, CLEsequences.file, CLEsequences.patient, COUNT(*), path, subfolder FROM CLEsequences LEFT JOIN CLEframes on CLEsequences.id == CLEframes.sequenceID LEFT JOIN CLEdatabases on CLEsequences.database == CLEdatabases.id '+constraint+' group by CLEframes.sequenceID');
        print('SELECT CLEsequences.id, CLEsequences.file, CLEsequences.patient, COUNT(*) FROM CLEsequences LEFT JOIN CLEframes on CLEsequences.id == CLEframes.sequenceID '+constraint+' group by CLEframes.sequenceID')
        result=self.c.fetchall()
        return result

    ##########################################################################
    ## getFileBatchFromFrameList:
    ## Store a CLEsequence entry in the database
    ##########################################################################

    def getFileBatchFromFrameList(self, frameList, basepath='../Data/'):
        self.c.execute('SELECT sequenceID, frameIdx, id FROM CLEframes ORDER by id ASC')
        allFrames=np.asarray(self.c.fetchall())
        indices=allFrames[:,2].searchsorted(frameList)

        allSequenceIds=np.unique(allFrames[indices,0])
        allSequenceIds.sort()
        self.c.execute('SELECT CLEsequences.id as id, CLEdatabases.path,  patient, file, subfolder  FROM CLEsequences LEFT JOIN CLEdatabases on CLEsequences.database == CLEdatabases.id ORDER by id ASC')
        allSequences=self.c.fetchall()
        allSequences=np.asarray(allSequences)
        sequenceIds = allSequences[:,0].astype(np.int16)
        indices_sequences=allSequences[:,0].astype(np.int16).searchsorted(allSequenceIds)
        batchArray=[]
        for seq in range(len(allSequenceIds)):
            newBatch = fileBatch()
            subf = '' 
            if ((allSequences[indices_sequences[seq],4] is not None)):
                if (len(allSequences[indices_sequences[seq],4])>0):
                    subf = allSequences[indices_sequences[seq],4] + os.path.sep
            
            newBatch.filename = (basepath + allSequences[indices_sequences[seq],1] + os.path.sep +
                                allSequences[indices_sequences[seq],2] + os.path.sep + subf + 
                                allSequences[indices_sequences[seq],3] + '.mkt')
            indices_frames=np.where(allFrames[indices,0]==allSequenceIds[seq])[0]
            newBatch.frames=allFrames[indices[indices_frames],1]
            batchArray.append(newBatch)

        return batchArray
    ##########################################################################
    ## storeEntry:
    ## Store a CLEsequence entry in the database
    ##########################################################################

    def storeEntry(self,entry):
        self.c.execute('UPDATE CLEsequences SET file=\''+entry.file+'\', fromIndex='+str(entry.fromIndex)+', toIndex='+str(entry.toIndex)+', fps='+str(entry.fps)+' WHERE id='+str(entry.DBidx));
        self.conn.commit();

    def __exit__(self ,type, value, traceback):
            self.conn.close()
