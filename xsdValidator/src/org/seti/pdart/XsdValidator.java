package org.seti.pdart;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.xml.transform.Source;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;
import javax.xml.validation.Validator;

import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

public class XsdValidator {
    class Args {
        private static final String usageMsg
            = "usage: java -jar XsdValidator.jar "
            + "<schemaFile> ... [ <xmlFile> | - ]";

        void usage(){
            System.err.println(usageMsg);
            System.exit(-1);
        }

        final ArrayList<Source> schemaDocuments = new ArrayList<Source>();
        final StreamSource xmlSource;

        Args(String[] args){
            final int argsLength = args.length;
            if (argsLength < 2){
                usage();
            }
            List<String> schemaFilepaths
                = Arrays.asList(args).subList(0, argsLength-1);
            for (String schemaFilepath : schemaFilepaths){
                File file = new File(schemaFilepath);
                StreamSource streamSource = new StreamSource(file);
                schemaDocuments.add(streamSource);
            }
            String xmlFilepath = args[argsLength - 1];

	    if ("-".equals(xmlFilepath)){
		xmlSource = new StreamSource(System.in);
	    } else {
		File file = new File(xmlFilepath);
		xmlSource = new StreamSource(file);
	    }
        }
    }

    private static final String LANG =  "http://www.w3.org/2001/XMLSchema";

    XsdValidator(String[] commandLineArgs) throws IOException, SAXException {
        Args args = new Args(commandLineArgs);
        SchemaFactory factory = SchemaFactory.newInstance(LANG);
        Source[] sources = args.schemaDocuments.toArray(new Source[0]);
        Schema schema = factory.newSchema(sources);
        Validator validator = schema.newValidator();
        validator.validate(args.xmlSource);
    }

    public static void main(String[] args) throws IOException, SAXException {
        try {
            new XsdValidator(args);
        } catch (SAXParseException e){
            /* TODO What is a system vs. public ID? */
            System.err.printf("\"%s\":%d:%d: %s", e.getSystemId(),
                              e.getLineNumber(), e.getColumnNumber(),
                              e.getMessage());
            System.err.println();
            System.exit(-1);
        }
    }
}
