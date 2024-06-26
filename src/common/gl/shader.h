// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_SHADER_H
#define VSNRAY_GL_SHADER_H 1

#include <GL/glew.h>

namespace visionaray
{
namespace gl
{

class shader
{
public:

    explicit shader(GLuint name = 0);
   ~shader();

    shader(shader const&) = delete;
    shader& operator=(shader&) = delete;

    void destroy();

    void reset(GLuint name = 0);

    GLuint get() const;

    void set_source(char const* source) const;
    void compile() const;
    bool check_compiled() const;

private:

    GLuint name_;
};

} // gl
} // visionaray

#endif // VSNRAY_GL_SHADER_H
