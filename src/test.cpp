#include <iostream>
#include <variant>
#include <glaze/glaze.hpp>
#include "Frame.h"
#include "utils.hpp"

using namespace dataframe;
using namespace dataframe::utils;

void df_test() {
  DataSlide slide1;
  slide1.add_param("hello1", 1);
  slide1.add_param("hello2", "fff");
  slide1.add_data("test_data");
  slide1.push_data("test_data", 10.2);

  DataSlide slide2;
  slide2.add_param("hello2", "fff");

  DataFrame frame;
  frame.add_slide(slide1);
  frame.add_slide(slide2);
  frame.add_param("test_param", 1);
  frame.add_metadata("test_metadata", 10);

  std::string buffer = glz::write_json(frame);
  buffer = glz::prettify(buffer);
  std::cout << "buffer = \n";
  std::cout << buffer << std::endl << std::endl;
  DataFrame frame2;
  auto pe = glz::read_json(frame2, buffer);
  std::string descriptive_error = glz::format_error(pe, buffer);
  std::cout << descriptive_error << std::endl;
  std::cout << "frame2: \n";
  std::cout << glz::prettify(glz::write_json(frame2)) << std::endl;
}

void basic_test() {
  var_t v{1.5};
  std::string buffer = glz::write_json(v);
  buffer = glz::prettify(buffer);
  std::cout << buffer << std::endl;
  auto pe = glz::read_json(v, buffer);
  buffer = std::visit(var_t_to_string(), v);
  std::cout << buffer << std::endl << std::endl;

  v = var_t{1.0};
  buffer = glz::write_json(v);
  buffer = glz::prettify(buffer);
  std::cout << buffer << std::endl;
  pe = glz::read_json(v, buffer);
  buffer = std::visit(var_t_to_string(), v);
  std::cout << buffer << std::endl << std::endl;

  v = var_t{"hello"};
  buffer = glz::write_json(v);
  buffer = glz::prettify(buffer);
  std::cout << buffer << std::endl;
  pe = glz::read_json(v, buffer);
  buffer = std::visit(var_t_to_string(), v);
  std::cout << buffer << std::endl << std::endl;


  Params params;
  params.emplace("k1", "v1");
  params.emplace("k2", 2.0);
  params.emplace("k3", 3.0);

  buffer = glz::write_json(params);
  buffer = glz::prettify(buffer);
  std::cout << "buffer = \n" << buffer << std::endl;

  Params params2;

  pe = glz::read_json(params2, buffer);
  std::string descriptive_error = glz::format_error(pe, buffer);
  std::cout << descriptive_error << std::endl;
  std::cout << "params2: \n" << glz::prettify(glz::write_json(params2)) << std::endl;

}

int main(int argc, char* argv[]) {
  basic_test();
  df_test();
}
