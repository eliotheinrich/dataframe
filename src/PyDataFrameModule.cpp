#include "PyDataFrame.hpp"

NB_MODULE(dataframe_bindings, m) {
	dataframe::init_dataframe(m);
}