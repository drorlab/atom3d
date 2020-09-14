#include <iostream>
using std::cout;
using std::endl;
#include <string>
#include "H5Cpp.h"
using namespace H5;

//const H5std_string FILE_NAME( "pdbbind.h5" );
//const H5std_string FILE_NAME( "../../data/ligand_binding_affinity/lba-split/lba_test_meta_10.h5" );
//const H5std_string DATASET_NAME( "metadata" );
const H5std_string FILE_NAME( "../../data/ligand_binding_affinity/lba-split/lba_test_0000_10.h5" );
const H5std_string DATASET_NAME( "structures" );


int main (void)
{
    /*
     * Open the specified file and the specified dataset in the file.
     */
    H5File* file = new H5File( FILE_NAME, H5F_ACC_RDONLY );
    Group* group = new Group( file -> openGroup( "structures" ) );
    cout<<"opened group"<<endl;
    /*
     * Access dataset in the group.
     */
    try // to determine if the dataset exists in the group
    {
        DataSet dataset = DataSet( group -> openDataSet( "block0_values" ) );
        /*
         * Get the class of the datatype that is used by the dataset.
         */
        H5T_class_t type_class = dataset.getTypeClass();
        cout<<type_class<<endl;	
	if( type_class == H5T_FLOAT )
        {
            cout << "Data set has FLOAT type" << endl;
            /*
             * Get the float datatype
             */
	    FloatType fltype = dataset.getFloatType();
	    /*
             * Get order of datatype and print message if it's a little endian.
	     */
            H5std_string order_string;
            H5T_order_t order = fltype.getOrder( order_string );
	    cout << order_string << endl;
            /*
	     * Get size of the data element stored in file and print it.
	     */
	    size_t size = fltype.getSize();
	    cout << "Data size is " << size << endl;
	}

	/*
         * Get dataspace of the dataset.
	 */
        DataSpace dataspace = dataset.getSpace();

        /*
	 * Get the number of dimensions in the dataspace.
	 */
	int rank = dataspace.getSimpleExtentNdims();

	/*
	 * Get the dimension size of each dimension in the dataspace and
	 * display them.
	 */
        hsize_t dims_out[2];
	int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
	cout << "rank " << rank << ", dimensions " <<
		(unsigned long)(dims_out[0]) << " x " <<
		(unsigned long)(dims_out[1]) << endl;

    } 
    catch( GroupIException not_found_error ) 
    {
        cout << " Dataset is not found." << endl;
    }
    cout << "Dataset is open" << endl;

    /*
     * Close the group and file.
     */
    delete group;
    delete file;

    return 0;  // successfully terminated
}

