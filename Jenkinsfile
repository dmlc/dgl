#!/usr/bin/env groovy

def init_git_submodule() {
    sh 'git submodule init'
    sh 'git submodule update'
}

def setup() {
    sh 'easy_install nose'
    init_git_submodule()
}

def build_dgl() {
    sh 'if [ -d build ]; then rm -rf build; fi; mkdir build'
    dir('python') {
        sh 'python3 setup.py install'
    }
    dir ('build') {
        sh 'cmake ..'
        sh 'make -j$(nproc)'
    }
}

def unit_test() {
    withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
        sh 'nosetests tests -v --with-xunit'
        sh 'nosetests tests/pytorch -v --with-xunit'
        sh 'nosetests tests/mxnet -v --with-xunit'
        sh 'nosetests tests/graph_index -v --with-xunit'
    }
}

def example_test(dev) {
    dir ('tests/scripts') {
        withEnv(["DGL_LIBRARY_PATH=${env.WORKSPACE}/build"]) {
            sh "./test_examples.sh ${dev}"
        }
    }
}

pipeline {
    agent none
    stages {
        stage('Lint Check') {
            agent {
                docker {
                    image 'lingfanyu/dgl-lint'
                }
            }
            stages {
                stage('CHECK') {
                    steps {
                        init_git_submodule()
                        sh 'tests/scripts/task_lint.sh'
                    }
                }
            }
        }
        stage('Build and Test on Pytorch') {
            parallel {
                stage('CPU') {
                    agent {
                        docker {
                            image 'lingfanyu/dgl-cpu'
                        }
                    }
                    stages {
                        stage('SETUP') {
                            steps {
                                setup()
                            }
                        }
                        stage('BUILD') {
                            steps {
                                build_dgl()
                            }
                        }
                        stage('UNIT TEST') {
                            steps {
                                unit_test()
                            }
                        }
                        stage('EXAMPLE TEST') {
                            steps {
                                example_test('CPU')
                            }
                        }
                    }
                    post {
                        always {
                            junit '*.xml'
                        }
                    }
                }
                stage('GPU') {
                    agent {
                        docker {
                            image 'lingfanyu/dgl-gpu'
                            args '--runtime nvidia'
                        }
                    }
                    stages {
                        stage('SETUP') {
                            steps {
                                setup()
                            }
                        }
                        stage('BUILD') {
                            steps {
                                build_dgl()
                            }
                        }
                        stage('UNIT TEST') {
                            steps {
                                unit_test()
                            }
                        }
                        stage('EXAMPLE TEST') {
                            steps {
                                example_test('GPU')
                            }
                        }
                    }
                    post {
                        always {
                            junit '*.xml'
                        }
                    }
                }
            }
        }
        stage('Build and Test on MXNet') {
            parallel {
                stage('CPU') {
                    agent {
                        docker {
                            image 'zhengda1936/dgl-mxnet-cpu'
                        }
                    }
                    stages {
                        stage('SETUP') {
                            steps {
                                setup()
                            }
                        }
                        stage('BUILD') {
                            steps {
                                build_dgl()
                            }
                        }
                        stage('UNIT TEST') {
                            steps {
                                unit_test()
                            }
                        }
                        stage('EXAMPLE TEST') {
                            steps {
                                example_test('CPU')
                            }
                        }
                    }
                    post {
                        always {
                            junit '*.xml'
                        }
                    }
                }
            }
        }
    }
}
