import java.util.*;
import java.io.*;
import java.util.stream.*;
import java.nio.file.*;

import compsci201.Ignore;

/**
 * Use ClassifyingModel instances to be trained on several
 * authors, then to attempt to identify "unknown" files
 * by calculating maximal likelihood estimation for
 * each unknown file.
 * 
 * @author Owen Astrachan
 * @version 1.0, for Compsci 201, Fall 2025, September 25
 * @version 2.0, Spring 2026
 */


public class Classifier201 {

    private Map<String, ClassifyingModel> myModels;  // dirname->model for dir
    private int mySize;                              // order of each model (same)
    private final double SMOOTHING = 0.1;
    private boolean DEBUG_FLAG = false;

    public Classifier201(){
        this(3);
    }

    public Classifier201(int size){
        myModels = new HashMap<>();
        mySize = size;
    }

    public Classifier201(int size, boolean debug){
        myModels = new HashMap<>();
        mySize = size;
        DEBUG_FLAG = debug;
    }

    /**
     * helper method for debugging when private DEBUG_FLAG is true
     * @param message
     */
    private void debug(String message) {
        if (DEBUG_FLAG) {
            System.out.print(message);
        }
    }

    /** 
     * @return the last /name in a /foo/bar/baz/name file/folder    
     */
    private String getDirName(String fullPath){
        int index = fullPath.lastIndexOf(File.separator);
        return fullPath.substring(index+1);
    }

    /**
     * Folder specified by dirName is a collection of folders, each representing
     * an author's works. Train a model for each folder/author, make models
     * accessible via other methods (modifies internal state).
     * @param dirName is the name of the folder containing author folders
     * @throws IOException if folders cannot be open/read
     */

    @Ignore
    public void trainAllAuthors(String dirName) throws IOException {
        Path root = Paths.get(dirName)
;        try (Stream<Path> paths = Files.walk(root)){
            List<Path> dirs = paths.filter(Files::isDirectory)
                                   .filter(path -> ! path.equals(root))
                                   .collect(Collectors.toList());

            debug(String.format("%s has %d subdirs\n",dirName,dirs.size()));
            for(Path each : dirs) {
                String trainDir = each.toAbsolutePath().toString();
                String dName = getDirName(trainDir);
                debug(String.format("training %14s\t",dName));
                               
                // second parameter to model constructor: true = memoize, false no memoize  
                ClassifyingModel model = new ClassifyingModel(mySize,true);//changed
                model.trainDirectory(trainDir);
                myModels.put(dName,model);

                
                debug(String.format("order %d with %d unique tokens, %d tokens\n",
                                    model.getOrder(),
                                    model.vocabularySize(),
                                    model.tokenSize()));
            }
        }      
    }

    public List<Map.Entry<String,Double>> findAllMatches(Path path) throws IOException{

        String text = Files.readString(path);
        Map<String,Double> record = new HashMap<>();

        for(String modelName: myModels.keySet()) {
            double start = System.nanoTime();
            double val = myModels.get(modelName)
                                 .calculateMatchProbability(text, SMOOTHING);
            double end = System.nanoTime();
            double time = (end-start)/1e9;
            debug(String.format("time: %1.2f for %s\n",time,modelName));
            record.put(modelName,val);
        }

        // sort map entries by log likelihood, with largest first, smallest last
        ArrayList<Map.Entry<String,Double>> all = new ArrayList<>(record.entrySet());
        Collections.sort(all, Map.Entry.comparingByValue(Comparator.reverseOrder()));
        return all;
    }

    /**
     * For each file in the folder with dirName, test against all models
     * and print information about which model is best (and what scores are)
     * by the call to findBestMatch
     * @param dirName is name of folder containing files to be identified
     * @throws IOException if reading files/folder fails
     */

    public void identifyAll(String dirName) throws IOException{
         try (Stream<Path> paths = Files.walk(Paths.get(dirName))){
            List<Path> files = paths.filter(Files::isRegularFile)
                                    .collect(Collectors.toList());

            for(Path each: files){
                identifyOne(each);
            }
         }
    }

    public void identifyOne(Path dirPath) throws IOException {
        List<Map.Entry<String,Double>> list = findAllMatches(dirPath);
        double max = list.get(0).getValue();
        String best = list.get(0).getKey();
        String fileName = dirPath.getFileName().toString();

        System.out.printf("*** %1.2f\t%s for %s\n",max,best,fileName);

        for(int k=0; k < list.size(); k++){
            System.out.printf("%1.2f\t%s\n",list.get(k).getValue(),list.get(k).getKey());
        }
        double ratio = Math.exp(max - list.get(1).getValue());
        System.out.printf("likelihood ratio: %1.2f\n",ratio);
    }

    /**
     * Train authors in first parameter/folder, identify works in second parameter/folder
     * @param trainingFolder contains folders, each for an author (train a model on each)
     * @param identifyFolder is a folder of unknown works, compare against all models
     * @throws IOException if reading folders/files fails
     */

    public void traindAndIdentify(String trainingFolder, String identifyFolder) throws IOException {
        trainAllAuthors(trainingFolder);
        identifyAll(identifyFolder);
    }

    public static void main(String[] args) throws IOException {
        Classifier201 classifier = new Classifier201(1,true); 
        String training = "data";
        String identify = "newauthors";
        //identify = "newauthors";
        classifier.traindAndIdentify(training,identify);
       
    }
}
