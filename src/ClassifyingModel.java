import java.io.*;
import java.util.*;
import java.util.regex.*;

/**
 * @author: ADD YOUR NAME HERE 201 STUDENT
 * @author: Owen Astrachan for Compsci 201
 */

public class ClassifyingModel extends BaseMarkovModel{

    private HashMap<List<String>, List<String>> myMap;
    private HashSet<String> myVocabulary;
    private Map<List<String>, Map<String,Integer>> myCache;
    private boolean myUseMemo;

    public ClassifyingModel(int size) {
        this(size,true);
        myCache = new HashMap<>();
    }

    public ClassifyingModel(int size, boolean memoize) {
        super(size);
        myUseMemo = memoize;
        myMap = new HashMap<>();
        myVocabulary = new HashSet<>();
        myCache = new HashMap<>();
    }

    private int tokenInContextCount(List<String> context, String token) {
        if (! myMap.containsKey(context)) return 0;
        if (myUseMemo && myCache.getOrDefault(context, new HashMap<>()).containsKey(token)){
            return myCache.get(context).get(token);
    }
        int count = 0;
        for(String s : myMap.get(context)) {
            if (s.equals(token)) {
                count += 1;
            }
        }
        if(myUseMemo && myCache.containsKey(context))
            myCache.get(context).put(token,count);
        else if(myUseMemo)
        {
            Map<String,Integer> test = new HashMap<>();
            test.put(token,count);
            myCache.put(context,(test));
        }
        return count;
    }

    /**
     * Use regular expression to tokenize rather
     * than split. Any alphabetic sequence followed by 
     * punctuation. So separation can be whitespace or number
     * for example.
     * @return list of tokens
     */

    @Override
    public List<String> tokenize(String text){
        List<String> tokens = new ArrayList<>();
        String includePunc = "[A-Za-z]+|[.,!?;:]";
        Pattern pattern = Pattern.compile(includePunc);
        Matcher matcher = pattern.matcher(text.toLowerCase());

        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    /**
     * Tokenize text and add <START>/<END> tags. Functionally
     * the same as BaseMarkovModel.updateWordSequence,
     * but that adds the padded text to an instance variable rather
     * than returning it. So code copied here
     * @param text to be processed
     * @return tokenized and pre/post padded sequence of tokens
     */
     private List<String> createTokenizedText(String text) {
        List<String> padded = new ArrayList<>();
        List<String> tokens = tokenize(text);

        for(int k=0; k < myModelSize; k++){
            padded.add("<START>");
        }
        padded.addAll(tokens);
        for(int k=0; k < myModelSize; k++){
            padded.add(END);
        }

        return padded;
    }

    /**
     * Return the log likelihood that text matches this trained model.
     * The text will be tokenized, then each n-gram/follow in text
     * "compared" in probabilistic way to the trained model's data.
     * Return the log probability of a match based on this comparison.
     * @param text is to be tokenized and matched against this model
     * @param smoother value used for Laplace smoothing
     * @return the normalized log probability of a match
     */

    public double calculateMatchProbability(String text, double smoother){
        
        List<String> padded = createTokenizedText(text);
        HashSet<List<String>> set = new HashSet<>();
        double probTotal = 0.0;
        int contextCount = 0;
        int nextCount = 0;

        for(int k=0; k < padded.size() - myModelSize; k++) {
            List<String> context = padded.subList(k, k+myModelSize);
            String next = padded.get(k+myModelSize);
            contextCount = myMap.getOrDefault(context, new ArrayList<>()).size();
            set.add(context);
            nextCount = tokenInContextCount(context, next);
            double prob = (nextCount + smoother)/(contextCount + smoother * vocabularySize()); // this will be replaced by appropriate calculations/values
            probTotal += Math.log(prob);
        }
        return probTotal/set.size();  // must be normalized
    }

    @Override
    public void processTraining(){
        // modify instance variable to track vocabulary
        myVocabulary.addAll(myWordSequence);
        
        for(int k=0; k < myWordSequence.size()-myModelSize; k++) {
            List<String> current = myWordSequence.subList(k, k+myModelSize);
            String next = myWordSequence.get(k+myModelSize);

            // additional instance variables may need to be initialized
            // when caching is implemented


            myMap.putIfAbsent(current,new ArrayList<>());
            myMap.get(current).add(next);    
        }
    }

    public int vocabularySize(){
        return myVocabulary.size();
    }

    public static void main(String[] args) throws IOException {
        ClassifyingModel mm = new ClassifyingModel(3);
        String dirName = "data/shakespeare";
        mm.trainDirectory(dirName);
        System.out.printf("trained model for %s, vocab size = %d, token size = %d\n",
                          dirName,mm.vocabularySize(),mm.tokenSize());
    }
}