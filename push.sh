#!/bin/bash
name=$1
cur=$PWD

{
  cd /local/crv/yiqing/SSCNav;
}
git add .  
git commit -m "${name}"  
git push origin  
cd "${cur}"
