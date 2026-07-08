#!/usr/bin/python3
#
# Namelist.py -- Version=1.0 (Aug/10/2008)
#                                            YoungMu Jeon
#                                            mailto: jeon@fusion.gat.com
# Namelist3.py --Version=1.1
#                                            Guoqiang Li@ASIPP
"""
A general/standard module to handle fortran namelist data with python


It is a general module to handle fortran namelist as new special dictionary. 
Since it inherits an "ordered-list" dictionary, 'oldict' class, one can use most of
all standard functionalities of python dictionary.
It provides basic methods such as creating, adding, updating, and reading/writing
from/to a file. Note that it can read/write almost all format of fortran namelist
excepr multi-dimensional array (>= 2-D). It supports 'var1 = 1, 2, 3', 'var1(3) =
10' and 'var2 = 5*1.2' so on. However it doesn't support 'var2(3,2) = 1.3' things,
because there is no way to figure out what dimensions the variable has originally.
Therefore if multi-dimension array is given in namelist, this module handles the 
variable as new one. This means 'var2' and 'var2(1,2)' are considered as two 
distinct variables.
In addition, it has a special variable, 'self.look'. This gives you the information
about what kind of multi-dimensional variables mentioned above are given from a 
namelist file. Since this is a dictionary, to see the variable names, one just
give a name of namelist block of interest.
Another useful feature is that although the python is sensitive to the character
case (upper or lower case), this module has an option 'case' to make user choose
the handling method of character case. If 'case' sets to 'upper' or 'lower', then
the namelist operation is case-insensitive (Really fortran style). The only
difference between 'upper' and 'lower' is the format of displaying or writing the
namelist. The other possible choice is to set 'case' to None. Then it works case-
sensitively (python style). Default of 'case' option is 'upper'.

IMPORTANT NOTES
(1) Values of any variables are considered as a list (array). Even if the value 
    is scalar, one must be put it as an array having one element.
(2) In case of boolean type variables, the value is displayed such as 'True' or
    'False' (it's python boolean type), but when writing to a file, it's converted 
    to '.true.' or '.false.' (fortran boolen type).

                                             -- Updated at April/15/2008
                                             -- YoungMu Jeon, jeon@fusion.gat.com

((Example))
## Creating
> from Namelist import Namelist   # Importing Namelist module

> obj =Namelist()                 # Build an empty namelist (object)
> obj2=Namelist("inone")          # Build a namelist with reading 'inone' file
> obj3=Namelist("inone",case=None)# Build a namelist with case-sensitivity
> obj3=Namelist(case='lower')     # Build a namelist with case-insensitivity and 
                                  # block's and variable's names are displayed with
                                  # lower character.

## Adding and Deleting
> obj["nml1"] = {}                # create new empty namelist block with a name of 'nml1'
> obj["nml1"]["time0"] = [2.0]    # add/update a variable 'time0' with a value of 2.0
                                  # into the namelist block 'nml1'. Note here that
                                  # to give a scalar, I used an array with one element
> obj["nml2"]["eqdsk"] = ['g096043.02200']
                                  # Build new namelist block 'nml2' and add new variable
                                  # 'eqdsk' with a value 'g096043.02200'
                                  # Note that if 'nml2' is not created before, then create
                                  # it first and insert 'eqdsk'.
> del obj["nml1"]["time0"]        # delete the variable 'time0' in 'namelist1' block

## Updating
> tmp = {'itti':0,'itte':1}       # To add/update multiple variables at once, one can
> obj["nml1"].update(tmp)         # use 'update()' method (it's a standard dictionary method)

> obj.update(obj2)                # One can update a namelist object with another namelist
                                  # object.

## Combining two namelists
> obj3 = obj1+obj2                # This '+' operation first copies obj1, then updates it
                                  # with obj2, and then returns it.

## Getting values
> obj["nml1"]                     # return one namelist block named as 'nml1'
> obj["nml1"]["time0"]            # return the value of 'time0' variable in 'nml1' block

## Reading an existing namelist file
> obj.read("test.dat")            # read namelist file 'test.dat'
> obj2 = Namelist('test.dat')     # Alternatively, you can read a file when you create
                                  # a new namelist object
  # NOTE : read() method calls clear() method internally. So all existing info is erased.
  
## Writing to a file
> obj.write("test.out")           # print out all namelist defined in obj
> obj.write("test.out","efit")    # print out a specific namelist block "efit"

"""
"""
UPDATE HISTORY:
- Updated for python3
                                                                  - 20200227 GL
- '_updateElements_()' upgraded, so it can handles 'rfon(1) = 3, 4, 5, 6'. This
  means all(almost) variations of inone file can be read now.     - 20080724 YMJ
- 'update()' upgraded. From now, one can combine two namelist object using this
  method.                                                         - 20080723 YMJ
- A method 'copy()' upgraded in order to copy real values, not references
                                                                  - 20080722 YMJ
- Important correction made to '__getBlock__()' function. Instead of using 
  '__CommentOut__()' and '__getBlocks__()' methods, it uses '__splitBlocks__()'
  from now.
                                                                  - 20080608 YMJ
- New option for CASE-SENSITIVITY added and some basic corrections made
                                                                  - 20080412 YMJ
- Huge improvements done. It's not compatible with old version. Now an instance 
  (object) of Namelist class can be treated as a standard python dictionary.
                                                                  - 20080215 YMJ
- Two problems fed back from JMP are solved. One is that this namelist retrieved
  value as "'d',',','h'" from given string "namep = 'd','h'". The other is that 
  when this namelist saves data into a file, the boolean value is shown as a string,
  so it can't be recognized as a boolean type. Both are fixed.    - 20080213 YMJ
- Added another symbol '$' or '&' to indicate namelist blocks. Therefore now it 
  support "&name ~ &", "$name ~ $", "&name ~ &end", "$name ~ $end", "&name ~ /",
  "$name ~ /"                                                     - 20080207 YMJ
- First created                                                   - 20080201 YMJ
"""
import sys
import re
import copy
#################################################################################
#
#  README
#
#    Namelist : A standard, general class to handle a fortran-namelist (reading,
#               writing, creating, and manipulating).
#
#    oldict   : OLDICT="Ordered and List-based DICTionary".
#               A root class of Namelist class.
#
#################################################################################

#===============================================================================#
#  DESCRIPTION
#
#  USAGE
#
#  KEY HISTORY
#  - Extended from 'odict' to 'oldict'                  - 20080801 YoungMu Jeon
#  - An option, 'case', to control the Case-Sensitivity added
#                                                       - 20080412 YoungMu Jeon
#  - First created                                      - 20080201 YoungMu Jeon
#===============================================================================#
#===============================================================================#
#
#  oldict class
#
#===============================================================================#
class oldict:
   """
   A generalized, standard "Ordered", "List-based" DICTionary.
   
   case   : if None   , case-sensitive (default)
            if "upper", case-insensitive and upper-character displayed
            if "lower", case-insensitive and lower-character displayed
   """
   def __init__(self,dict=None,case=None,**kwargs):
       self.data = {}
       self.okeys= []
       self.case = case
       if dict:
          self.update(dict)
       if len(kwargs):
          self.update(kwargs)
   
   def __repr__(self):
       str = "{"
       for key in self.okeys:
           str += "'%s': %s, " %(key,self[key])
       if  len(str)==1: str += "}"
       else: str = str[:-2]+"}"
       
       return str
   
   def __len__(self): return len(self.data)

   def __getitem__(self, key):
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if key in self.okeys: return self.data[key]
       else: print("key not found"); return

   def __setitem__(self, key, item):
       if  self.case:
           if self.case.lower()=="upper": key = key.upper()
           if self.case.lower()=="lower": key = key.lower()
       
       if   hasattr(item,"items") : self.data[key] = oldict(item,case=self.case)
       elif type(item) != type([]): self.data[key] = [item] # make a list
       else                       : self.data[key] = item
       
       if not key in self.okeys: self.okeys.append(key)
   
   def __delitem__(self, key):
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if key in self.okeys:
          self.okeys.remove(key)
          del self.data[key]
   
   def __add__(self,other):
       new_oldict = self.copy()
       new_oldict.update(other)
       return new_oldict
   
   def clear(self): self.data.clear(); self.okeys=[]
   def copy(self):
       """
       This copy is not same with intrinsic copy() of python. Intrinsic copy() just copies
       its value, so that for a dictionary or a list it copies its reference, not value. 
       Here this copy() is designed to copy its corresponding value. 
       Note that in most cases it works well, but if it has a dictionary as its value, then
       it copies its reference. (It's impossible to generalize this functionality to a
       dictionary-type value, because there is no way to figure out how many levels that
       dictionary has.)
       """
       import copy
       return copy.deepcopy(self)

   def keys(self): return self.okeys[:]

   def items(self):
       item = []
       for k in self.okeys: item.append((k,self[k]))
       return item

   def has_key(self, key):
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if key in self.okeys: return True
       return False

   def update(self, dict=None, **kwargs):
       """ This update means adding new ones and replacing existing ones. """
       # For dict
       if dict:
          for k,v in dict.items():
             if self.case:
                if self.case.lower()=="upper": k = k.upper()
                if self.case.lower()=="lower": k = k.lower()
             if   hasattr(v,"items") : self.data[k] = oldict(v,case=self.case)
             elif type(v) != type([]): self.data[k] = [v]
             else                    : self.data[k] =  v
             if not k in self.okeys: self.okeys.append(k)
       # For kwargs
       if len(kwargs):
          for k, v in kwargs.items():
             if self.case:
                if self.case.lower()=='upper': k = k.upper()
                if self.case.lower()=='lower': k = k.lower()
             if   hasattr(v,"items") : self.data[k] = oldict(v,case=self.case)
             elif type(v) != type([]): self.data[k] = [v]
             else                    : self.data[k] =  v
             if not k in self.okeys: self.okeys.append(k)

   def get(self,key,notkey=None):
       """
       Get the value corresponding to the key. If the given key is not
       recognized, then the notkey's value is returned
       """
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if   key in self.okeys: return self[key]
       elif notkey: return notkey
       else: return None

   def pop(self,key,notkey=None):
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if key in self.okeys:
          self.okeys.remove(key)
          return self.data.pop(key)
       elif notkey: return notkey
       else: return None
   
   def popitem(self):
       if len(self.okeys):
          key = self.okeys.pop()
          val = self.data.pop(key)
          return (key,val)
       else: return (None,None)
   
   def __contains__(self, key):
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       return self.data._contains_(key)



#===============================================================================#
#
#  NAMELIST class
#
#===============================================================================#
class Namelist(oldict):
   """
   A standard, general class to handle a fortran-namelist (reading, writing,
   creating, and manipulating).
   """
   #-------------------------------------------------------------------#
   def __init__(self,filename="",string="",nml=None,case="upper"):
       """
       A constructor. If argument 'filename' is given, then it will read that
       file automatically. The read data is stored to a dictionary, whose key
       is names of each namelist and value is another dictionary, whose key 
       is names of variables and value is their value list.
       """
       # initialize
       self.case     = case
       self.head     = ""
       self.tail     = ""
       self.look     = oldict(case=self.case)
       self.namelist = oldict(case=self.case)
       
       self.data     = self.namelist.data
       self.okeys    = self.namelist.okeys
       
       if nml: self.update(nml)
       
       if   (filename != ""): self.read(filename=filename)
       elif (string   != ""): self.read(string=string)
       else                 : pass
  
   #-------------------------------------------------------------------#
   def __repr__(self):
       """ display the namelist contents in a certain form """
       str0 = "" 
       for key,value in self.namelist.items():
          str0  += "[[%s]]\n" %(key)
          strtmp = ""
          for k,v in value.items():
             strtmp += "%s = %s\n" %(k,v)
          if strtmp == "": strtmp = "{}\n"
          str0 += strtmp
       if str0=="": str0 = "{}"
       return str0
      
   #-------------------------------------------------------------------#
   def __setitem__(self, key, item):
       """ """
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if hasattr(item,"keys"):
          self.data[key] = oldict(item,case=self.case)
       else:
          print ("[Error] in '%s'.__setitem__() : value must be a dictionary" %(__name__) )
          return
       
       if not key in self.okeys: self.okeys.append(key)
   
   def __getitem__(self, key):
       """
       Note that this method of Namelist.py has special feature, that is,
       if user tries to access non-existing key (namelist blockname), then
       that key is created automatically and return empty dictionary.
       """
       if self.case:
          if self.case.lower()=="upper": key = key.upper()
          if self.case.lower()=="lower": key = key.lower()
       if key in self.okeys: return self.data[key]
       else:
          self.__setitem__(key,{})
          return self.__getitem__(key)

   #-------------------------------------------------------------------#
   def __add__(self,other):
       """ sum two objects and return new combined one """
       new_nml = self.copy()
       new_nml.update(other)
       return new_nml
      
   #-------------------------------------------------------------------#
   def copy(self): return copy.deepcopy(self)

   #-------------------------------------------------------------------#
   def clear(self):
       """ Reset namelist """
       # Make inherited functions work by making same object variables
       # <-- Should I have to define variables with same names as parent class???
       self.head     = ""
       self.tail     = ""
       self.look     = oldict(case=self.case)
       self.namelist = oldict(case=self.case)
       
       self.data     = self.namelist.data
       self.okeys    = self.namelist.okeys
      
   #-------------------------------------------------------------------#
   def update(self,nml):
       """
       Update the elements. Only self.namelist and self.look are updated
       ('head' and 'tail', and 'case' attribute are kept with original
       values.). 
       """
       if  not isinstance(nml,Namelist):
           print("Error : Must be an instance of Namelist")
           return None
       
       for k in nml.keys():
           if  self.case:
               if self.case.lower()=="upper": k = k.upper()
               if self.case.lower()=="lower": k = k.lower()
           self[k].update(nml[k])
           
   #-------------------------------------------------------------------#
   def getHead(self):
       """ Return a string of the head """
       return self.head[:]
      
   def setHead(self,str):
       """ Set a head """
       self.head = str[:]
      
   #-------------------------------------------------------------------#
   def read(self,filename="",string=""):
       """
       Read a namelist from a file. Returns a dictionary whose key is names 
       of each namelist and value is another dictionary, whose key is names
       of variables and value is their value list.
       
       INPUTS:
         filename   : filename to be read
         string     : string to be read
       """
       self.clear()
       
       # Convert to a String
       if filename != "":
          try:
             f = open(filename,"r")
             lines = f.read(); f.close()
          except:
             print("[Error] in %s.read() : Given file not found" %(__name__))
             return
       elif (string != ""): lines = string[:]
       else:
          print("[Error] in %s.read() : proper input required" %(__name__))
          return 
       
       # Find head-comments and take it off
       res = re.search(r"^\s*?[&$](?=(\w)+?\s)",lines,re.M)
       self.head = lines[ :res.span()[0] ]
       lines = lines[ res.span()[1]-1: ]
       
       # Split each namelist blocks and resolve each variable assignment
       blocks = self.__splitBlocks__(lines)
       for k,v in blocks.items():
           varDict,look = self.__getAssignments__(v[0])
           self.look[k] = look[:]
           self.namelist[k] = varDict
   
   def __splitBlocks__(self,str):
       """
       Split whole string into each strings of namelist blocks. Also figures out the 
       heading and tailing comments separately. Comments contained in each lines also
       removed. Return value is a dictionary whose key is a name of namelist-block and
       whose value is a string of each namelist block.
       """
       idx_start = []   # a list of starting point index
       idx_end   = []   # a list of ending   point index
       dict      = oldict(case=self.case)
                        # a set of pairs whose key is the name of namelist-block
                        # and whose value is a block string
       
       # Find out rough positions of starting of namelist-block
       iter = re.finditer(r"^\s*?[&$](?=(\w)+?\s)",str,re.M)
       for match in iter:
           dtmp = match.span()
           if str[dtmp[1]:dtmp[1]+3].lower() == 'end': continue # if [$&]end, then skip
           idx_start.append(dtmp[1])
           idx_end.append(dtmp[0])
       
       # Find out rough positions of ending of namelist-block
       # - Old value of idx_end indicates each start-index of matched patterns
       nblock = len(idx_start)
       if nblock == 0: return []
       for i in range(0,nblock-1): idx_end[i] = idx_end[i+1]
       idx_end[nblock-1] = len(str)
       
       # Figure out the names of namelist-blocks and find out the exact end of block
       for i in range(0,nblock):
           str1 = str[ idx_start[i] : idx_end[i] ]
           
           # Name of namelist-block
           res  = re.match(r"\w+?(?=\s)",str1)
           name = res.group()
           idx_start[i] = idx_start[i]+len(name)
           str1 = str1[len(name):]
           
           # Find the end of namelist block
           # - I assumed that only '[$&/](end)?$' case is valid for the end of block.
           iend,iend2 = None, None
           res = re.search(r"\s*?[$&/](?=(end)?$)",str1,re.I|re.M)
           if res: (iend,iend2) = res.span()
           if iend==None: # No valid end-symbol found
              print("Error: Not standard end-symbol of namelist block")
              sys.exit()
           # therefore ... returned blocks are ...
           str2 = str1[:iend]
           pat  = re.compile(r"[!;].*?$",re.M)
           dict[name.upper()] = [ pat.sub(" ",str2) ] # Comments taken away
           
           # Find the tail comments
           if i == nblock-1:
              self.tail = str1[iend2:]
              if self.tail[:3].lower() == "end": self.tail = self.tail[3:]
           
       return dict
      
   #-------------------------------------------------------------------#
   def __getAssignments__(self,str):
       """
       Get a dictionary of (variable, value) from an input string. Input 'str'
       is a multiline string which contains a series of 'variable = values'.
       First it splits 'str' to substrings of 'variable = values', and then
       calls 'splitAssing()' function to get its variable name and values.
       Return value is a dictionary whose key is each variable name and value
       is each value such as [ 'variable1 = value','variable2 =value2'].
       """
       pattern = re.compile(r"""
                                 # '.' = any character except newline. In DOTALL, any character
                                 # '+' = 1 or more matching
                                 # '*' = 0 or more matching
                                 # '?' = 0 or 1 matching
                  [a-zA-Z0-9_]+? # any alphanumeric character and '_'
                  [0-9(),]*?
                  \s*? = \s*?    # search ' = ' things
                  .*?            # any characters
                  (?=            # '(?=...)' means that matched if following string matches
                                 #           with '...'
                  [a-zA-Z0-9_]+?
                  [0-9(),]*?
                  \s*? = |$      # '$' = end of the string or just before newline
                  )
                 """,re.DOTALL|re.VERBOSE)
       
       # First, let's delete a space between "(" and ")".
       fiter = re.compile(r"[(].*?[)]").finditer(str)
       spans = []; str_new = ""
       for match in fiter: spans.append(match.span())
       for t in range(0,len(spans)):
           start   = spans[t][0]; end = spans[t][1]
           tgt_str = str[start:end]
           tgt_str = re.compile(r"\s").sub("",tgt_str)
           if t==0: str_new = str_new + str[              :spans[0][0] ] + tgt_str
           else   : str_new = str_new + str[ spans[t-1][1]:start       ] + tgt_str
       if ( len(spans) != 0 ): str_new = str_new + str[ spans[-1][1]: ]
       if ( len(spans) == 0 ): str_new = str[:]
       
       # Split whole string to blocks
       dict = oldict()
       dict["__look__"] = []
       for pat in pattern.findall(str_new):
           variable,value = self.__splitAssign__(pat)
           dict = self.__updateElements__(variable,value,dict) # updating dict
       
       look = dict["__look__"][:]
       dict.pop("__look__")
       return dict,look
   
   #-------------------------------------------------------------------#
   def __splitAssign__(self,str):
       """
       Get a variable name and its value from a 'str'. Input 'str' has a 
       form, 'variable = values'. Values can be one of followings; scalar,
       1d-array, string, string-array. To indicate a string, one can use
       "'" or '"' symbol. To distinguish each elements, one can use white-
       space or ",". Return value is a dictionary whose key is its variable
       name and value is its value (single value or list).
       """
       variable,value = str.split("=")
       variable = variable.strip().upper()
       if re.compile(r"\n").search(value): # if newline, then substitute by " "
          value = re.compile(r"\n").sub(" ",value)
       
       if re.compile(r"(\'.*?\')|(\".*?\")").search(value): # if string or character
          value1 = re.compile(r"""['].*?[']""").findall(value)
          value2 = re.compile(r"""["].*?["]""").findall(value)
          value  = []
          for val in value1: value.append(val[1:-1])
          for val in value2: value.append(val[1:-1])
       elif re.compile(r"([.]true[.])|([.]false[.])|([.]t[.])|([.]f[.])|t|f|true|false",\
                       re.I).search(value): # boolean
          value0 = re.compile(r",").sub(" ",value) # substitute ',' with ' '
          value0 = value0.split()
          value  = []
          for val in value0:
             val0 = val.lower()
             if val0 in [".true.", "true", ".t.", "t"]: value.append(True)
             else                                     : value.append(False)
       else: # if numeric
          if re.compile(r",").search(value): # if it split with ',', then
             value = re.compile(r",").sub(" ",value) # substitute ',' with ' '
          value0 = value.split()
          value1 = []
          # type conversion
          if re.compile(r"[.eE]").search(value): # if float type,
             for val in value0:
                if re.compile(r"[*]").search(val):
                   res = val.split("*")
                   n = int(res[0])
                   for t in range(0,n): value1.append(float(res[1]))
                else:
                   value1.append(float(val))
          else:                                  # integer
             for val in value0:
                if re.compile(r"[*]").search(val):
                   res = val.split("*")
                   n = int(res[0])
                   for t in range(0,n): value1.append(int(res[1]))
                else:
                   value1.append(int(val))
          value = value1[:]
       
       return variable, value
   
   #-------------------------------------------------------------------#
   def __updateElements__(self,variable,value,dict):
       """
       Update special assignments for single element. For example, 'var(3)=3.0'.
       Before calling this routine, 'var(3)' and 'var' were considered as distinct
       different variables. 
       """
       varlist = dict.keys()
       if re.search(r"[(][0-9]+?[)]",variable): # for accessing 1-d element
          varname = re.match(r"[a-zA-Z0-9_]+?(?=[(])",variable).group()
          varindx = int(variable[len(varname)+1:-1])
       
          if (varname in varlist):
             val  = dict[varname]
             nlen = len(val)
             if nlen < varindx: # if accessing non-existing element
                ntmp = varindx-nlen
                if(type(val[0])==type(2)): val = val + [0  ]*ntmp
                else                     : val = val + [0.0]*ntmp
             n = len(value)
             val[varindx-1:varindx-1+n] = value[:]
             dict[varname] = val[:]
          else: # if new variable
             if(type(value[0])==type(2)): val = [0  ]*(varindx-1)
             else                       : val = [0.0]*(varindx-1)
             dict[varname] = val + value
             
       else: # if trivial variable or multi-dimension
          if (variable in varlist):
             val = dict[variable]
             if len(val) < len(value): dict[variable] = value[:]
             else                    : dict[variable][:len(value)] = value[:]
             
             if "(" in variable: # multi-dimension assignment
                dict["__look__"].remove(variable)
                dict["__look__"].append(variable)
          else:
             dict[variable] = value[:]
             
             if "(" in variable: # multi-dimension assignment
                dict["__look__"].append(variable)
       
       return dict

   #-------------------------------------------------------------------#
   def write(self,fname,nmlname="",status="w",maxCol=72,indent=2,split=" "):
       """
       Write namelist dictionary to a given file. If 'nmlname' is given, 
       then it will print out given namelist block to file 'fname'. If not,
       it will print out all namelist blocks. 'maxCol' and 'indent' are
       optionally arguments for better formatting.
       
       INPUTS:
         fname      : file name for saving
         nmlname    : specific name of namelist block
         status     : file openning status. "w"(new), "a"(append) available
         maxCol     : maximum column number for fancy formatting
         indent     : indentation size for fancy formatting
         split      : split symbol between each values.
       """
       if not ((status=="w") or (status=="a")):
          print( "[Error] in %s.write() : Wrong arguments'" %(__name__) )
          return
       
       try:
          f = open(fname,status)
       except:
          print( "[Error] in %s.write() : File cannot created" %(__name__) )
          return
       
       lines = []
       
       if(nmlname==""): # All namelist blocks will be printed
          if(self.head != ""): lines.append(self.head[:])
          for k,v in self.namelist.items():
             lines.append("&"+k+"\n")
             lines += self.__namelist2str__(v,maxCol,indent,split)
             lines.append("/\n\n")
          if(self.tail != ""): lines.append(self.tail[:])
       
       else: # One particular namelist block will be printed
          if self.case:
             if self.case.lower()=="upper": nmlname = nmlname.upper()
             if self.case.lower()=="lower": nmlname = nmlname.lower()
          try:
             nmls = self.namelist[nmlname]
          except: 
             print( "[Error] in %s.write() : Not defined namelist name" %(__name__) )
          
          lines.append("&"+nmlname+"\n")
          lines += self.__namelist2str__(nmls,maxCol,indent,split)
          lines.append("/\n\n")
       
       f.writelines(lines)
       f.close()

   #-------------------------------------------------------------------#
   def __namelist2str__(self,nml,maxCol=72,indent=3,split=" "):
       """
       Convert a namelist dictionary to string lists
       
       INPUTS:
         nml        : namelist dictionary. This must be one namelist block, 
                      so its keys are variable names and its values are real
                      data.
         maxCol     : maximum column number for fancy formatting
         indent     : indentation size for fancy formatting
         split      : split symbol between each values.
       """
       import re
       
       space   = " "*indent # indentation
       nmlStr  = []
       for name,value in nml.items():
          data   = value[:]
          line0  = space+name+" = "
          iempty = False
          while (not iempty):
             if (data==[]):
                nmlStr.append(line0+"\n")
                iempty = True
             else:
                n0 = len(line0)
                n1 = len(str(data[0])+split)
                if (n0+n1 > maxCol):
                   nmlStr.append(line0+split+"\n")
                   line0 = space*2
                else:
                   if   type(data[0]) == type("")  :  # string type
                        line0 += "'" + str(data[0]) + "'" + split
                   elif type(data[0]) == type(True):  # Boolean
                        if data[0] == True : line0 += ".true."  + split
                        if data[0] == False: line0 += ".false." + split
                   else: # numeric type
                        line0 += str(data[0]) + split
       
                   if len(data)>=2: data = data[1:]
                   else           : data = []
       
       return nmlStr   
      
#################################################################################
if __name__ == "__main__":

   if(len(sys.argv) != 2):
      print( "  Usages : namelist.py filename" )
      sys.exit()
      
   filename = sys.argv[1]
   obj = Namelist(filename)
   print(obj)
   print("Tail-Comments : ", obj.tail)
   

